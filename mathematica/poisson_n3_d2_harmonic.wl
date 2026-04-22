(* ::Package:: *)
(*
  poisson_n3_d2_harmonic.wl
  ---------------------------------------------------------------------------
  Independent Mathematica oracle for the harmonic (r^2) planar 3-body
  Poisson algebra.

  Reproduces the cumulative-rank dimension sequence
      [3, 6, 13, 15]
  at d=2, N=3 with H_ij = T_i + T_j + r_ij^2.

  The harmonic potential closes at dimension 15 and is the structural
  opposite of the singular potentials (1/r, 1/r^2, 1/r^3, log) which all
  give [3, 6, 17, 116] and grow without bound.

  Run:    wolframscript -file mathematica/poisson_n3_d2_harmonic.wl
  Output: mathematica/results/n3_d2_harmonic.json
  ---------------------------------------------------------------------------
*)

ClearAll["Global`*"];

scriptDir = If[Head[$InputFileName] === String && $InputFileName =!= "",
               DirectoryName[$InputFileName],
               Directory[]];
Get[FileNameJoin[{scriptDir, "poisson_n3_d2_engine.wl"}]];

maxLevel = 4;
pot      = "harmonic";
expected = {3, 6, 13, 15, 15};

resultsDir = FileNameJoin[{scriptDir, "results"}];
If[!DirectoryQ[resultsDir], CreateDirectory[resultsDir]];
outFile = FileNameJoin[{resultsDir, "n3_d2_harmonic.json"}];

Print["============================================================"];
Print["  Mathematica oracle: planar 3-body harmonic Poisson algebra"];
Print["  Wolfram Language ", $Version];
Print["  N=", nBodies, "  d=", dSpatial, "  potential=", pot,
      "  maxLevel=", maxLevel];
Print["============================================================"];

Print[];
result = buildAlgebra[pot, maxLevel];

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
  "potential"       -> pot,
  "max_level"       -> maxLevel,
  "expected"        -> expected,
  "result"          -> toAssoc[result]
|>;

Export[outFile, jsonOut, "JSON"];

Print[];
Print["============================================================"];
Print["  SUMMARY"];
Print["============================================================"];
Module[{r = result["cumulative_rank"], hit, exp},
  exp = Take[expected, UpTo[Length[r]]];
  hit = r === exp;
  Print[pot, ":  cumulative_rank = ", r,
        "   expected ", exp,
        "   ", If[hit, "MATCH", "MISMATCH"]];
];

Print[];
Print["Results saved to: ", outFile];
