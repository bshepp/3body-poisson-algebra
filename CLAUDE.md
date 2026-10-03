# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is deliberately minimal. The project itself is described in [README.md](README.md) (papers, results, layout);
current state and the deployment log live in [docs/project_status.md](docs/project_status.md) and
[docs/session_log.md](docs/session_log.md). Read those before non-trivial work. This file only records
infrastructure items that are easy to lose track of.

## Website bucket is public — pending OAC lockdown (noted 2026-10-02)

The site `nbody.briansheppard.com` is deployed with `aws s3 sync website/ s3://nbody-briansheppard-com/`
(see `docs/atlas_compute_workorder.md`) followed by a CloudFront invalidation. `infra/userdata_sphere.sh`
also writes results into this bucket (`S3_BUCKET`); the lockdown below does not affect that — it only removes
anonymous *read* access, and the EC2 role keeps writing through IAM.

`nbody-briansheppard-com` is served by CloudFront `E3AHN5BEM2KUCH` (`nbody.briansheppard.com`) through the **S3 website endpoint**
(`nbody-briansheppard-com.s3-website-us-east-1.amazonaws.com`). That origin type only works while the bucket has
Block Public Access **off** and a public-read bucket policy, so anyone can also fetch objects straight from S3,
bypassing CloudFront. As of 2026-10-02 this is one of three buckets in the account still public
(`alcubierre.briansheppard.com`, `nbody-briansheppard-com`, `storm-water-simple-frontend`); every other site
bucket is private behind an Origin Access Control. Account-level Block Public Access is waiting on these three.

**To lock it down** (about 10 minutes; a brief blip on cache misses is acceptable, nothing else changes):

1. Create an Origin Access Control: `aws cloudfront create-origin-access-control --origin-access-control-config Name=nbody-briansheppard-com-oac,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3`
2. Update distribution `E3AHN5BEM2KUCH` (`get-distribution-config` → edit → `update-distribution --if-match <ETag>`):
   origin `DomainName` → `nbody-briansheppard-com.s3.us-east-1.amazonaws.com`; remove `CustomOriginConfig`; add
   `S3OriginConfig: {"OriginAccessIdentity": ""}` and `OriginAccessControlId: <oac id>`; set `DefaultRootObject: index.html`;
   add custom error responses for **403 and 404 → `/index.html`, 200** (S3 REST answers 403 for missing keys).
3. Replace the bucket policy with a single statement: `Allow s3:GetObject` to `Principal: {"Service": "cloudfront.amazonaws.com"}`
   on `arn:aws:s3:::nbody-briansheppard-com/*` with `Condition StringEquals AWS:SourceArn = arn:aws:cloudfront::290318879194:distribution/E3AHN5BEM2KUCH`.
   Do not write a policy that still contains the `Principal: "*"` statement — the Claude Code auto-mode classifier blocks that as a weakening, and it is unnecessary if step 2 runs first.
4. `aws s3api put-public-access-block --bucket nbody-briansheppard-com --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true`
   then `aws s3api delete-bucket-website --bucket nbody-briansheppard-com`.
5. `aws cloudfront wait distribution-deployed --id E3AHN5BEM2KUCH`, then verify `https://nbody.briansheppard.com/`, a static asset, and a bad path (should return `index.html` with 200)
   and confirm `https://s3.us-east-1.amazonaws.com/nbody-briansheppard-com/index.html` now returns **403**.
6. `docs/project_status.md` notes a deploy is held pending review; the lockdown is independent of that and can go first.

Reference implementation: `F:\dark-forest-labs-projects\ubomw-site` — commit `cb6bf21` ("move S3 behind CloudFront OAC") and `scripts/deploy_s3.sh` there. Migrated 2026-10-02 with the exact steps below.
