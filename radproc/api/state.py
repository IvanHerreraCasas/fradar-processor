# radproc/api/state.py
import asyncio

# Define shared state variables here
image_update_queue = asyncio.Queue()

# You could add other shared state later if needed, e.g.:
# active_sse_clients = set()
# background_task_status = {}