(* ::Package:: *)
(*
  poisson_n3_d2.wl
  ---------------------------------------------------------------------------
  Independent Mathematica oracle for the planar 3-body Poisson algebra.

  Reproduces the cumulative-rank dimension sequence
      [3, 6, 17, 116]
  for both 1/r and 1/r^2 pairwise potentials at d=2, N=3.

  Mirrors the Python `nbody/symbolic_rank_nbody.py` filtration exactly so
  the level-by-level cumulative ranks match the published numbers.

  Phase-space variables:    {x_i, y_i, px_i, py_i}, i = 1..3
  Auxiliary variables:      u_ij = 1/r_ij                (i < j)
  H_ij = (px_i^2 + py_i^2)/2 + (px_j^2 + py_j^2)/2 - F(u_ij)

  Run:    wolframscript -file mathematica/poisson_n3_d2.wl
  Output: mathematica/results/n3_d2_dimseq.json
  ---------------------------------------------------------------------------
*)

ClearAll["Global`*"];

scriptDir = If[Head[$InputFileName] === String && $InputFileName =!= "",
               DirectoryName[$InputFileName],
               Directory[]];
Get[FileNameJoin[{scriptDir, "poisson_n3_d2_engine.wl"}]];

maxLevel   = 3;
potentials = {"1/r", "1/r^2"};

resultsDir = FileNameJoin[{scriptDir, "results"}];
If[!DirectoryQ[resultsDir], CreateDirectory[resultsDir]];
outFile = FileNameJoin[{resultsDir, "n3_d2_dimseq.json"}];

Print["============================================================"];
Print["  Mathematica oracle: planar 3-body Poisson algebra"];
Print["  Wolfram Language ", $Version];
Print["  N=", nBodies, "  d=", dSpatial, "  maxLevel=", maxLevel];
Print["============================================================"];

results = <||>;
Do[
  Print[];
  results[pot] = buildAlgebra[pot, maxLevel];
  ,
  {pot, potentials}
];

toAssoc[a_Association] := Association @ KeyValueMap[
  Function[{k, v}, ToString[k] -> If[Head[v] === Association, toAssoc[v], v]],
  a
];

jsonOut = <|
  "wolfram_version" -> $Version,
  "system_id"       -> $SystemID,
  "timestamp_utc"   -> DateString[Now, "ISODateTime"],
  "n_bodies"        -> nBodies,
  "d_spatial"       -> dSpatial,
  "max_level"       -> maxLevel,
  "results"         -> Association @ KeyValueMap[
                         Function[{k, v}, ToString[k] -> toAssoc[v]],
                         results]
|>;

Export[outFile, jsonOut, "JSON"];

Print[];
Print["============================================================"];
Print["  SUMMARY"];
Print["============================================================"];
expected = {3, 6, 17, 116};
Do[
  Module[{r = results[pot]["cumulative_rank"], hit, exp},
    exp = Take[expected, UpTo[Length[r]]];
    hit = r === exp;
    Print[pot, ":  cumulative_rank = ", r,
          "   expected ", exp,
          "   ", If[hit, "MATCH", "MISMATCH"]];
  ],
  {pot, potentials}
];

Print[];
Print["Results saved to: ", outFile];
(* ::Package:: *)
(*
  poisson_n3_d2.wl
  ---------------------------------------------------------------------------
  Independent Mathematica oracle for the planar 3-body Poisson algebra.

  Reproduces the cumulative-rank dimension sequence
      [3, 6, 17, 116]
  for both 1/r and 1/r^2 pairwise potentials at d=2, N=3.

  This is a verification script, not a performance engine. It mirrors the
  Python `nbody/symbolic_rank_nbody.py` filtration convention exactly so
  that the level-by-level cumulative ranks match the published numbers.

  Phase-space variables:    {x_i, y_i, px_i, py_i}, i = 1..3
  Auxiliary variables:      u_ij = 1/r_ij                (i < j)
  H_ij = (px_i^2 + py_i^2)/2 + (px_j^2 + py_j^2)/2 - F(u_ij)
    where F(u) = u  for 1/r,  u^2 for 1/r^2,  u^3 for 1/r^3.

  Chain rule on u_ij (treating it as an opaque symbol):
      d u_ij / d x_k = -(x_i - x_j) u_ij^3 (delta_ki - delta_kj)
      d u_ij / d y_k = -(y_i - y_j) u_ij^3 (delta_ki - delta_kj)
      d u_ij / d p_k = 0

  Filtration (matches Python engine):
    Level 0: H_12, H_13, H_23
    Level 1: { {H_a, H_b} : a < b in level 0 }
    Level k: bracket each (level k-1) frontier element with each prior
             generator (level 0..k-1), modulo a frozenset-pair dedupe.

  Run:    wolframscript -file mathematica/poisson_n3_d2.wl

  Output: mathematica/results/n3_d2_dimseq.json
  ---------------------------------------------------------------------------
*)

ClearAll["Global`*"];

(* ---- Configuration ---------------------------------------------------- *)
nBodies   = 3;
dSpatial  = 2;
maxLevel  = 3;
potentials = {"1/r", "1/r^2"};

scriptDir  = If[Head[$InputFileName] === String && $InputFileName =!= "",
                DirectoryName[$InputFileName],
                Directory[]];
resultsDir = FileNameJoin[{scriptDir, "results"}];
If[!DirectoryQ[resultsDir], CreateDirectory[resultsDir]];

outFile = FileNameJoin[{resultsDir, "n3_d2_dimseq.json"}];

(* ---- Symbol setup ----------------------------------------------------- *)
xSym[i_]    := ToExpression["x"  <> ToString[i]];
ySym[i_]    := ToExpression["y"  <> ToString[i]];
pxSym[i_]   := ToExpression["px" <> ToString[i]];
pySym[i_]   := ToExpression["py" <> ToString[i]];
uSym[i_, j_] := ToExpression["u" <> ToString[i] <> ToString[j]];

qVars   = Flatten @ Table[{xSym[i], ySym[i]},  {i, nBodies}];
pVars   = Flatten @ Table[{pxSym[i], pySym[i]}, {i, nBodies}];
uVars   = Flatten @ Table[uSym[i, j], {i, nBodies}, {j, i+1, nBodies}];
allVars = Join[qVars, pVars, uVars];

(* ---- Chain rule for u_ij = 1/r_ij ------------------------------------ *)
dUdQ[i_, j_, var_] := Which[
  var === xSym[i],  -(xSym[i] - xSym[j]) uSym[i, j]^3,
  var === xSym[j],   (xSym[i] - xSym[j]) uSym[i, j]^3,
  var === ySym[i],  -(ySym[i] - ySym[j]) uSym[i, j]^3,
  var === ySym[j],   (ySym[i] - ySym[j]) uSym[i, j]^3,
  True, 0
];

(* Position derivative with chain rule through every u_ij *)
PDq[expr_, qvar_] := Module[{base, chain},
  base  = D[expr, qvar];
  chain = Sum[
    D[expr, uSym[i, j]] * dUdQ[i, j, qvar],
    {i, nBodies}, {j, i+1, nBodies}
  ];
  base + chain
];

(* Momentum derivative — no chain rule needed *)
PDp[expr_, pvar_] := D[expr, pvar];

(* ---- Poisson bracket {f, g} = sum (df/dq dg/dp - df/dp dg/dq) -------- *)
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

HPair[i_, j_, pot_] := kineticI[i] + kineticI[j] +
                       potentialTerm[uSym[i, j], pot];

H0[pot_] := Flatten @
  Table[HPair[i, j, pot], {i, nBodies}, {j, i+1, nBodies}];

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
    exprs, levels, ranks, n0, computedPairs,
    frontier, nExist, workItems, expr, tStart, tLevel, lv, i, j, pair,
    rawPerLevel
  },

  Print["[", pot, "] generating algebra up to level ", maxLv];
  tStart = AbsoluteTime[];

  exprs  = H0[pot];
  n0     = Length[exprs];
  levels = ConstantArray[0, n0];
  computedPairs = Internal`Bag[];
  ranks       = <|0 -> exprListRank[exprs]|>;
  rawPerLevel = <|0 -> n0|>;
  Print["  L0: ", n0, " gens, cumulative rank = ", ranks[0]];

  (* Level 1: brackets among level-0 pairs *)
  If[maxLv >= 1,
    tLevel = AbsoluteTime[];
    Do[
      Do[
        AppendTo[exprs,  PB[exprs[[i]], exprs[[j]]]];
        AppendTo[levels, 1];
        Internal`StuffBag[computedPairs, {i, j}];
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

  (* Levels 2..maxLv: bracket frontier with all prior, dedupe by frozenset *)
  Do[
    tLevel    = AbsoluteTime[];
    frontier  = Flatten @ Position[levels, lv - 1];
    nExist    = Length[exprs];

    workItems = {};
    pairsSoFar = Internal`BagPart[computedPairs, All];
    pairsSet = Association[# -> True & /@ pairsSoFar];
    Do[
      Do[
        If[i === j, Continue[]];
        pair = If[i < j, {i, j}, {j, i}];
        If[KeyExistsQ[pairsSet, pair], Continue[]];
        pairsSet[pair] = True;
        Internal`StuffBag[computedPairs, pair];
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

(* ---- Main ------------------------------------------------------------ *)
Print["============================================================"];
Print["  Mathematica oracle: planar 3-body Poisson algebra"];
Print["  Wolfram Language ", $Version];
Print["  N=", nBodies, "  d=", dSpatial, "  maxLevel=", maxLevel];
Print["============================================================"];

results = <||>;
Do[
  Print[];
  results[pot] = buildAlgebra[pot, maxLevel];
  ,
  {pot, potentials}
];

(* ---- Save JSON & summary --------------------------------------------- *)
toAssoc[a_Association] := Association @ KeyValueMap[
  Function[{k, v}, ToString[k] -> If[Head[v] === Association, toAssoc[v], v]],
  a
];

jsonOut = <|
  "wolfram_version" -> $Version,
  "system_id"       -> $SystemID,
  "timestamp_utc"   -> DateString[Now, "ISODateTime"],
  "n_bodies"        -> nBodies,
  "d_spatial"       -> dSpatial,
  "max_level"       -> maxLevel,
  "results"         -> Association @ KeyValueMap[
                         Function[{k, v}, ToString[k] -> toAssoc[v]],
                         results]
|>;

Export[outFile, jsonOut, "JSON"];

Print[];
Print["============================================================"];
Print["  SUMMARY"];
Print["============================================================"];
Do[
  Module[{r = results[pot]["cumulative_rank"], expected = {3, 6, 17, 116}, hit},
    hit = Take[r, UpTo[Length[expected]]] === Take[expected, UpTo[Length[r]]];
    Print[pot, ":  cumulative_rank = ", r,
          "   expected ", Take[expected, UpTo[Length[r]]],
          "   ", If[hit, "MATCH", "MISMATCH"]];
  ],
  {pot, potentials}
];

Print[];
Print["Results saved to: ", outFile];
