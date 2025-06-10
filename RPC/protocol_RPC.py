import json

class RPCProtocol:
    @staticmethod
    def encode_request(method, params):
        request = {
            "method": method,
            "params": params
        }
        return (json.dumps(request) + '\n').encode()

    @staticmethod
    def decode_request(data):
        try:
            request = json.loads(data)
            return request.get("method"), request.get("params")
        except Exception:
            return None, None

    @staticmethod
    def encode_response(result=None, error=None):
        response = {
            "result": result,
            "error": error
        }
        return (json.dumps(response) + '\n').encode()

    @staticmethod
    def decode_response(data):
        try:
            response = json.loads(data)
            return response.get("result"), response.get("error")
        except Exception:
            return None, "Invalid response format"
