import asyncio
from protocol_RPC import RPCProtocol

HOST = '127.0.0.1'
PORT = 9999

functions = {
    "add": lambda a, b: a + b,
    "echo": lambda msg: msg
}

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[Server] Connected from {addr}")

    while True:
        try:
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
            break

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
