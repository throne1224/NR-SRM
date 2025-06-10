import asyncio
import sys
from protocol_RPC import RPCProtocol

HOST = '127.0.0.1'
PORT = 9999

class RPCClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    async def call_async(self, method, *params):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        request = RPCProtocol.encode_request(method, params)
        writer.write(request)
        await writer.drain()

        data = await reader.readline()
        writer.close()
        await writer.wait_closed()

        result, error = RPCProtocol.decode_response(data.decode())
        if error:
            raise Exception(f"RPC Error: {error}")
        return result

    def call_sync(self, method, *params):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():

            raise RuntimeError("同步接口 call_sync() 不能在异步环境中调用，请使用 await call_async()")
        else:
            return asyncio.run(self.call_async(method, *params))


async def main():
    client = RPCClient(HOST, PORT)

    res_async = await client.call_async('add', 10, 20)
    print(f"[Client] 异步调用 add(10, 20) 结果: {res_async}")

    res_sync = await client.call_async('echo', 'Hello RPC')
    print(f"[Client] 同步调用 echo('Hello RPC')（通过 async） 结果: {res_sync}")

if __name__ == "__main__":
    asyncio.run(main())
