#!/usr/bin/env python3
"""Create HypeVault scraping queue (requires AWS credentials)."""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    region = sys.argv[1] if len(sys.argv) > 1 else "ap-south-1"
    client = boto3.client("sqs", region_name=region)
    name = "HypeVault-Scraping-Queue"
    try:
        resp = client.create_queue(
            QueueName=name,
            Attributes={
                "VisibilityTimeout": "120",
                "ReceiveMessageWaitTimeSeconds": "20",
            },
        )
        print(resp.get("QueueUrl"))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "QueueAlreadyExists":
            url = client.get_queue_url(QueueName=name)
            print(url["QueueUrl"])
        else:
            raise


if __name__ == "__main__":
    main()
