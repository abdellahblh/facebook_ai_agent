"""Debug script to inspect and fix stuck Redis Stream messages.

Usage:
    python debug_queue.py              # Show stream status
    python debug_queue.py --process    # Force-process all stuck messages
    python debug_queue.py --clear      # Delete the stream and start fresh (DEV ONLY)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.cache import streams
import redis.asyncio as aioredis


async def show_status(redis, stream, group):
    """Show detailed stream status."""
    print("\n=== Redis Stream Status ===\n")
    
    # Stream length
    length = await redis.xlen(stream)
    print(f"Stream length: {length}")
    
    # Consumer group info
    try:
        groups = await redis.xinfo_groups(stream)
        for g in groups:
            gdict = dict(g)
            name = gdict.get(b"name", b"").decode()
            pending = gdict.get(b"pending", 0)
            consumers = gdict.get(b"consumers", 0)
            last_id = gdict.get(b"last-delivered-id", b"?").decode()
            print(f"\nGroup: {name}")
            print(f"  Pending: {pending}")
            print(f"  Consumers: {consumers}")
            print(f"  Last delivered ID: {last_id}")
            
            # Show pending entries
            if pending > 0:
                print(f"\n  Pending entries:")
                pending_info = await redis.xpending_range(stream, group, min="-", max="+", count=100)
                for entry in pending_info:
                    entry_id = entry[b"message_id"].decode()
                    consumer = entry[b"consumer"].decode()
                    idle = entry[b"time_since_delivered"]
                    deliveries = entry[b"times_delivered"]
                    print(f"    {entry_id} - consumer: {consumer} - idle: {idle}ms - deliveries: {deliveries}")
    except Exception as e:
        print(f"\nNo consumer group found: {e}")
    
    # Show first few messages in stream
    print(f"\nFirst 5 messages in stream:")
    try:
        messages = await redis.xrange(stream, count=5)
        for entry_id, fields in messages:
            print(f"  {entry_id.decode()}: {fields}")
    except Exception as e:
        print(f"  Error: {e}")


async def process_stuck_messages(redis, stream, group, consumer="debug-consumer"):
    """Force-process all messages in the stream."""
    print("\n=== Processing Stuck Messages ===\n")
    
    # Ensure group exists
    await streams.ensure_consumer_group(redis, stream, group)
    
    # Read ALL messages (including already delivered ones)
    print("Reading messages with '>' cursor (undelivered)...")
    entries = await streams.read_next(redis, group, consumer, stream=stream, count=100, block_ms=100)
    
    if not entries:
        print("No undelivered messages found with '>' cursor.")
        print("\nTrying to reclaim from PEL...")
        entries = await streams.claim_pending(redis, group, consumer, stream=stream, min_idle_ms=0, count=100)
    
    if not entries:
        print("No messages to process.")
        return
    
    print(f"Found {len(entries)} messages to process:")
    
    # Import worker to process messages
    from app.worker import _process_queued
    from app.db import engine as db_engine
    from app.config import get_settings
    
    settings = get_settings()
    await db_engine.init_engine(settings.async_database_url)
    
    for entry_id, payload in entries:
        print(f"\nProcessing {entry_id}...")
        try:
            await _process_queued(redis, db_engine.SessionFactory, stream, group, entry_id, payload)
            print(f"  ✓ Processed and ACKed")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


async def clear_stream(redis, stream, group):
    """Delete the stream and consumer group (DEV ONLY)."""
    print("\n=== Clearing Stream ===\n")
    print(f"Deleting stream: {stream}")
    await redis.delete(stream)
    print("Stream deleted. Next app start will recreate it.")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", action="store_true", help="Process stuck messages")
    parser.add_argument("--clear", action="store_true", help="Clear the stream (DEV ONLY)")
    args = parser.parse_args()
    
    settings = get_settings()
    redis = aioredis.from_url(settings.redis_url, decode_responses=False)
    stream = settings.redis_stream
    group = settings.redis_stream_group
    
    try:
        if args.clear:
            await clear_stream(redis, stream, group)
        elif args.process:
            await process_stuck_messages(redis, stream, group)
        else:
            await show_status(redis, stream, group)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
