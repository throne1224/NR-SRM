import asyncio
from protocol import xor_encrypt_decrypt, get_trust_id

HOST = '127.0.0.1'
PORT = 8888

async def handle_client(reader, writer):
    try:
        encrypted_data = await reader.read(1024)
        decrypted = xor_encrypt_decrypt(encrypted_data.decode(), get_trust_id())

        if decrypted.decode() == get_trust_id():
            writer.write(b'OK')
            await writer.drain()
        else:
            writer.write(b'REJECT')
            await writer.drain()
            writer.close()
            return
    except Exception as e:
        print(f"Error: {e}")
        writer.close()

    data = await reader.read(1024)
    message = data.decode()
    writer.write(f"Echo: {message}".encode())
    await writer.drain()
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    asyncio.run(main())
