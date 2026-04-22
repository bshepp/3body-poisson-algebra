(* ::Package:: *)
(*
  poisson_n3_d2_engine.wl
  ---------------------------------------------------------------------------
  Shared Poisson-bracket / Lie-closure engine used by both
    poisson_n3_d2.wl              (L=3 sanity for 1/r and 1/r^2)
    poisson_n3_d2_l4_backup.wl    (L=4 1/r backup if HF Jobs fails)

  Defines:
    PB[f, g]               — Poisson bracket with chain rule on u_ij
    H0[pot]                — list of pairwise Hamiltonians
    exprListRank[exprs]    — exact rank over Q via SparseArray + MatrixRank
    buildAlgebra[pot, k]   — Lie closure to depth k, returns Association
                             with cumulative_rank and timing

  See poisson_n3_d2.wl for the documentation header.
  ---------------------------------------------------------------------------
*)

(* ---- Configuration shared across runners ----------------------------- *)
nBodies   = 3;
dSpatial  = 2;

(* ---- Symbol setup ----------------------------------------------------- *)
xSym[i_]    := ToExpression["x"  <> ToString[i]];
ySym[i_]    := ToExpression["y"  <> ToString[i]];
pxSym[i_]   := ToExpression["px" <> ToString[i]];
pySym[i_]   := ToExpression["py" <> ToString[i]];
uSym[i_, j_] := ToExpression["u" <> ToString[i] <> ToString[j]];

qVars   = Flatten @ Table[{xSym[i],  ySym[i]},  {i, nBodies}];
pVars   = Flatten @ Table[{pxSym[i], pySym[i]}, {i, nBodies}];
uVars   = Flatten @ Table[uSym[i, j], {i, nBodies}, {j, i + 1, nBodies}];
allVars = Join[qVars, pVars, uVars];

(* ---- Chain rule for u_ij = 1/r_ij ------------------------------------ *)
dUdQ[i_, j_, var_] := Which[
  var === xSym[i],  -(xSym[i] - xSym[j]) uSym[i, j]^3,
  var === xSym[j],   (xSym[i] - xSym[j]) uSym[i, j]^3,
  var === ySym[i],  -(ySym[i] - ySym[j]) uSym[i, j]^3,
  var === ySym[j],   (ySym[i] - ySym[j]) uSym[i, j]^3,
  True, 0
];

PDq[expr_, qvar_] := D[expr, qvar] + Sum[
  D[expr, uSym[i, j]] * dUdQ[i, j, qvar],
  {i, nBodies}, {j, i + 1, nBodies}
];

PDp[expr_, pvar_] := D[expr, pvar];

PB[f_, g_] := Together[
  Sum[
    PDq[f, qVars[[k]]] * PDp[g, pVars[[k]]] -
    PDp[f, pVars[[k]]] * PDq[g, qVars[[k]]],
    {k, Length[qVars]}
  ]
];

(* ---- Hamiltonians ----------------------------------------------------- *)
kineticI[i_] := (pxSym[i]^2 + pySym[i]^2) / 2;

potentialTerm[u_, "1/r"]   := -u;
potentialTerm[u_, "1/r^2"] := -u^2;
potentialTerm[u_, "1/r^3"] := -u^3;

(* Harmonic uses r_ij^2 directly in the q-coordinates, no u_ij needed.
   Sign matches the Python `exact_growth.py` convention:
   H_ij = T_i + T_j + g * r_ij^2  (g = 1).  Special-cased in HPair below. *)
rSq2D[i_, j_] := (xSym[i] - xSym[j])^2 + (ySym[i] - ySym[j])^2;

HPair[i_, j_, "harmonic"] := kineticI[i] + kineticI[j] + rSq2D[i, j];
HPair[i_, j_, pot_]       := kineticI[i] + kineticI[j] +
                             potentialTerm[uSym[i, j], pot];

H0[pot_] := Flatten @
  Table[HPair[i, j, pot], {i, nBodies}, {j, i + 1, nBodies}];

(* ---- Coefficient extraction & rank ----------------------------------- *)
exprToCoeffRules[expr_] := Module[{e},
  e = Expand[Together[expr]];
  CoefficientRules[e, allVars]
];

exprListRank[exprs_] := Module[
  {coeffsList, allMonos, monoIndex, entries, mat},
  coeffsList = exprToCoeffRules /@ exprs;
  allMonos   = DeleteDuplicates @ Flatten[coeffsList[[All, All, 1]], 1];
  monoIndex  = AssociationThread[allMonos -> Range[Length[allMonos]]];
  entries = Flatten[
    Table[
      With[{rules = coeffsList[[r]]},
        ({r, monoIndex[#[[1]]]} -> #[[2]]) & /@ rules
      ],
      {r, Length[coeffsList]}
    ],
    1
  ];
  mat = SparseArray[entries, {Length[coeffsList], Length[allMonos]}];
  MatrixRank[mat]
];

(* ---- Lie closure (mirrors Python filtration) ------------------------- *)
buildAlgebra[pot_, maxLv_] := Module[{
    exprs, levels, ranks, n0, pairsSet,
    frontier, nExist, workItems, expr, tStart, tLevel, lv, i, j, k, pair,
    rawPerLevel
  },

  Print["[", pot, "] generating algebra up to level ", maxLv];
  tStart = AbsoluteTime[];

  exprs       = H0[pot];
  n0          = Length[exprs];
  levels      = ConstantArray[0, n0];
  pairsSet    = <||>;
  ranks       = <|0 -> exprListRank[exprs]|>;
  rawPerLevel = <|0 -> n0|>;
  Print["  L0: ", n0, " gens, cumulative rank = ", ranks[0]];

  If[maxLv >= 1,
    tLevel = AbsoluteTime[];
    Do[
      Do[
        AppendTo[exprs,  PB[exprs[[i]], exprs[[j]]]];
        AppendTo[levels, 1];
        pairsSet[{i, j}] = True;
        ,
        {j, i + 1, n0}
      ],
      {i, 1, n0}
    ];
    rawPerLevel[1] = Length[exprs] - n0;
    ranks[1] = exprListRank[exprs];
    Print["  L1: +", rawPerLevel[1], " raw, cumulative rank = ", ranks[1],
          "  [", Round[AbsoluteTime[] - tLevel, 0.01], "s]"];
  ];

  Do[
    tLevel    = AbsoluteTime[];
    frontier  = Flatten @ Position[levels, lv - 1];
    nExist    = Length[exprs];

    workItems = {};
    Do[
      Do[
        If[i === j, Continue[]];
        pair = If[i < j, {i, j}, {j, i}];
        If[KeyExistsQ[pairsSet, pair], Continue[]];
        pairsSet[pair] = True;
        AppendTo[workItems, {i, j}];
        ,
        {j, 1, nExist}
      ],
      {i, frontier}
    ];

    rawPerLevel[lv] = Length[workItems];
    Print["  L", lv, ": frontier=", Length[frontier],
          " existing=", nExist, " candidates=", Length[workItems]];

    Do[
      {i, j} = workItems[[k]];
      expr = PB[exprs[[i]], exprs[[j]]];
      AppendTo[exprs,  expr];
      AppendTo[levels, lv];
      ,
      {k, Length[workItems]}
    ];

    ranks[lv] = exprListRank[exprs];
    Print["    cumulative rank = ", ranks[lv],
          "  [", Round[AbsoluteTime[] - tLevel, 0.01], "s]"];
    ,
    {lv, 2, maxLv}
  ];

  <|
    "potential"          -> pot,
    "n_bodies"           -> nBodies,
    "d_spatial"          -> dSpatial,
    "max_level"          -> maxLv,
    "elapsed_s"          -> Round[AbsoluteTime[] - tStart, 0.01],
    "cumulative_rank"    -> Table[ranks[lv],       {lv, 0, maxLv}],
    "raw_gens_per_level" -> Table[rawPerLevel[lv], {lv, 0, maxLv}],
    "total_generators"   -> Length[exprs]
  |>
];
