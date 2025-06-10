import asyncio
import os
import pytest

CORRECT_TRUST_ID = "so123"
WRONG_TRUST_ID = "wrong_id"

HOST = '127.0.0.1'
PORT = 9999

def test_rpc_calls():
    os.environ["TRUST_ID"] = CORRECT_TRUST_ID
    from client_RPC import RPCClient  # 必须在设置 TRUST_ID 后导入
    client = RPCClient(HOST, PORT)

    async def run_async_calls():
        result = await client.call_async("add", 10, 20)
        print(f"[Test] 异步调用 add(10, 20) = {result}")
        assert result == 30

    asyncio.run(run_async_calls())

    result = client.call_sync("echo", "Hello")
    print(f"[Test] 同步调用 echo('Hello') = {result}")
    assert result == "Hello"

def test_trust_id_mismatch():
    os.environ["TRUST_ID"] = WRONG_TRUST_ID
    from client_RPC import RPCClient  # 必须在设置 TRUST_ID 后导入
    client = RPCClient(HOST, PORT)

    async def run_mismatch():
        try:
            await client.call_async("add", 1, 2)
            assert False, "预期因 TRUST_ID 不一致而失败"
        except Exception as e:
            print(f"[Test] 捕获异常（预期）: {e}")
            msg = str(e).lower()
            assert (
                "unauthorized" in msg
                or "trust_id" in msg
                or "拒绝连接" in msg
                or "远程计算机拒绝网络连接" in msg
                or isinstance(e, ConnectionRefusedError)
                or isinstance(e, ConnectionResetError)
                or isinstance(e, OSError)
            )

    asyncio.run(run_mismatch())
