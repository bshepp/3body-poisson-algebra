# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It is deliberately minimal. The project itself is described in [README.md](README.md) (papers, results, layout);
current state and the deployment log live in [docs/project_status.md](docs/project_status.md) and
[docs/session_log.md](docs/session_log.md). Read those before non-trivial work. This file only records
infrastructure items that are easy to lose track of.

## Website hosting: private S3 bucket behind CloudFront OAC (locked down 2026-11-01)

The site `nbody.briansheppard.com` is CloudFront `E3AHN5BEM2KUCH` reading the private bucket
`nbody-briansheppard-com` through Origin Access Control `E21EP7BLNZ2LVA` (`nbody-briansheppard-com-oac`),
REST origin `nbody-briansheppard-com.s3.us-east-1.amazonaws.com`. Block Public Access is on, the bucket
policy grants `s3:GetObject` only to `cloudfront.amazonaws.com` for this distribution, and S3 website
hosting is removed. Missing keys (S3 answers 403 or 404) are served as `/404.html` with status 404.
`infra/userdata_sphere.sh` still writes results into this bucket through the EC2 role; unaffected.

Consequences for deploys and content:
- Deploy is unchanged: `aws s3 sync website/ s3://nbody-briansheppard-com/` then
  `aws cloudfront create-invalidation --distribution-id E3AHN5BEM2KUCH --paths "/*"`.
- Do NOT re-add a `Principal: "*"` bucket policy or website hosting.
- The REST origin does not serve directory index documents: `/foo/` is a 404, only `/foo/index.html`
  works. The site had no directory-style links at lockdown time; keep it that way or add a CloudFront
  Function to rewrite them.
- Objects are no longer fetchable at `https://s3.us-east-1.amazonaws.com/nbody-briansheppard-com/...` (403).

Reference implementation: `F:\dark-forest-labs-projects\ubomw-site` (commit `cb6bf21`, `scripts/deploy_s3.sh`).
