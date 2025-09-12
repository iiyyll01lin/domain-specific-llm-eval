from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import status

class ServiceError(Exception):
    def __init__(self, error_code: str, message: str, http_status: int = status.HTTP_400_BAD_REQUEST):
        self.error_code = error_code
        self.message = message
        self.http_status = http_status

async def service_error_handler(request: Request, exc: ServiceError):
    trace_id = getattr(request.state, 'trace_id', 'n/a')
    return JSONResponse(status_code=exc.http_status, content={
        'error_code': exc.error_code,
        'message': exc.message,
        'trace_id': trace_id
    })

async def generic_error_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, 'trace_id', 'n/a')
    return JSONResponse(status_code=500, content={
        'error_code': 'internal_error',
        'message': 'Internal server error',
        'trace_id': trace_id
    })
