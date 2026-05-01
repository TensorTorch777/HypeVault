# AWS Infrastructure Skill

When building AWS-related code:

- Always use boto3 with explicit region
- S3 keys follow pattern: {category}/{brand}/{listing_id}/{filename}
- SQS messages include: product_name, listing_id, timestamp
- All S3 URLs are pre-signed with 7-day expiry
- ECS task memory: 2048 MB, CPU: 1024
- Always handle ClientError from boto3
