import os
import asyncio
import subprocess
import pytest

HOST = '127.0.0.1'
PORT = 8888

@pytest.mark.asyncio
async def test_trust_match():
    os.environ['TRUST_ID'] = 'so123'

    server = subprocess.Popen(['python', 'server.py'])

    await asyncio.sleep(1)

    os.environ['TRUST_ID'] = 'so123'
    client = subprocess.run(['python', 'client.py'], capture_output=True, text=True)

    server.terminate()

    assert 'Received: Echo:' in client.stdout

@pytest.mark.asyncio
async def test_trust_mismatch():
    os.environ['TRUST_ID'] = 's123'

    server = subprocess.Popen(['python', 'server.py'])
    await asyncio.sleep(1)

    os.environ['TRUST_ID'] = 'wrong_id'
    client = subprocess.run(['python', 'client.py'], capture_output=True, text=True)

    server.terminate()

    assert 'Server rejected connection.' in client.stdout
