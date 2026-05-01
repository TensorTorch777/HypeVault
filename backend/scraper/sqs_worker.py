"""SQS long-poll worker: scrape, persist `price_snapshots` for a listing."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from database import PriceSnapshot
from scraper.aggregator import aggregate_prices

_log = logging.getLogger(__name__)

WAIT_SEC = 20
VISIBILITY = 60


def _client():
    return boto3.client(
        "sqs",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


async def _flush_listing_prices(listing_id: UUID, bundle: dict) -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            try:
                await session.execute(delete(PriceSnapshot).where(PriceSnapshot.listing_id == listing_id))

                def add_rows(platform: str, rows: list[dict]) -> None:
                    for row in rows or []:
                        price = row.get("lowest_ask")
                        if price is None:
                            continue
                        session.add(
                            PriceSnapshot(
                                listing_id=listing_id,
                                platform=platform,
                                price=float(price),
                                currency="USD",
                                delivery_estimate=row.get("estimated_delivery"),
                                seller_rating=str(row.get("seller_rating") or "N/A"),
                            )
                        )

                add_rows("stockx", bundle.get("stockx") or [])
                add_rows("chrono24", bundle.get("chrono24") or [])
                add_rows("ebay", bundle.get("ebay") or [])
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    finally:
        await engine.dispose()


async def process_message_body(body: str) -> None:
    try:
        data = json.loads(body)
        product_name = data.get("product_name")
        listing_id_raw = data.get("listing_id")
        if not product_name or not listing_id_raw:
            _log.warning("sqs_invalid_message: %s", body)
            return
        listing_id = UUID(str(listing_id_raw))
        bundle = await aggregate_prices(str(product_name))
        await _flush_listing_prices(listing_id, bundle)
    except Exception as exc:
        _log.exception("process_message_failed: %s", exc)
        raise


def run_worker_forever() -> None:
    if not settings.sqs_queue_url:
        _log.error("SQS_QUEUE_URL not configured; worker exiting")
        return
    sqs = _client()
    _log.info("sqs_worker_started queue=%s", settings.sqs_queue_url)
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=settings.sqs_queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=WAIT_SEC,
                VisibilityTimeout=VISIBILITY,
            )
            for msg in resp.get("Messages", []):
                receipt = msg["ReceiptHandle"]
                try:
                    asyncio.run(process_message_body(msg["Body"]))
                    sqs.delete_message(QueueUrl=settings.sqs_queue_url, ReceiptHandle=receipt)
                except ClientError as exc:
                    _log.exception("sqs_delete_failed: %s", exc)
                except Exception as exc:
                    _log.exception("sqs_message_failed: %s", exc)
        except ClientError as exc:
            _log.exception("sqs_receive_failed: %s", exc)
        except Exception as exc:
            _log.exception("sqs_loop_failed: %s", exc)


if __name__ == "__main__":
    run_worker_forever()
