import asyncio
import os
from protocol_RPC import RPCProtocol

HOST = '127.0.0.1'
PORT = 9999

TRUST_ID = os.environ.get("TRUST_ID", "")

functions = {
    "add": lambda a, b: a + b,
    "echo": lambda msg: msg
}

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[Server] Connected from {addr}")

    try:
        # 第一步：读取客户端发来的 TRUST_ID
        trust_id_line = await reader.readline()
        client_trust_id = trust_id_line.decode().strip()
        print(f"[Server] Received TRUST_ID from client: {client_trust_id}")

        if client_trust_id != TRUST_ID:
            print(f"[Server] Unauthorized client: {addr}")
            writer.write(b"Unauthorized\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        else:
            writer.write(b"OK\n")
            await writer.drain()

        # 第二步：正式进入 RPC 请求处理流程
        while True:
            data = await reader.readline()
            if not data:
                break

            method, params = RPCProtocol.decode_request(data.decode())
            if method in functions:
                try:
                    result = functions[method](*params)
                    response = RPCProtocol.encode_response(result=result)
                except Exception as e:
                    response = RPCProtocol.encode_response(error=str(e))
            else:
                response = RPCProtocol.encode_response(error="Method not found")

            writer.write(response)
            await writer.drain()
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[Server] Connection closed: {addr}")

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"[Server] RPC Server running on {HOST}:{PORT}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
