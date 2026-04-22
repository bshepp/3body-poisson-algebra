(* ::Package:: *)
(*
  poisson_n3_d2_l4_backup.wl
  ---------------------------------------------------------------------------
  Backup oracle for the L=4 N=3 d=2 1/r question, in case the HF Jobs
  cpu-xl run OOMs or times out (Phase D2, jobs 69e820f1cd8c002f31e0140b
  and 69e820f6ac288e522d8f075c).

  Same engine as poisson_n3_d2.wl, but only the 1/r potential and only
  the L=4 result. May take hours on a workstation.

  Run:    wolframscript -file mathematica/poisson_n3_d2_l4_backup.wl

  Output: mathematica/results/n3_d2_l4_1r.json

  Will print incremental cumulative ranks at L=0, 1, 2, 3, 4 — so you
  can confirm [3, 6, 17, 116, ?] and the L=4 entry will resolve the
  open question of whether 116 closes the algebra or extends.
  ---------------------------------------------------------------------------
*)

ClearAll["Global`*"];

(* Reuse the engine *)
scriptDir = If[Head[$InputFileName] === String && $InputFileName =!= "",
               DirectoryName[$InputFileName],
               Directory[]];
Get[FileNameJoin[{scriptDir, "poisson_n3_d2_engine.wl"}]];

(* Override config and run only 1/r at L=4 *)
maxLevel = 4;
result   = buildAlgebra["1/r", maxLevel];

outFile = FileNameJoin[{scriptDir, "results", "n3_d2_l4_1r.json"}];
Export[outFile, <|
  "wolfram_version" -> $Version,
  "system_id"       -> $SystemID,
  "timestamp_utc"   -> DateString[Now, "ISODateTime"],
  "result"          -> Association @ KeyValueMap[
                         Function[{k, v}, ToString[k] -> v], result]
|>, "JSON"];

Print["L=4 1/r cumulative_rank = ", result["cumulative_rank"]];
Print["Results saved to: ", outFile];
