from fastapi import FastAPI
from services.common.middleware import TraceMiddleware
from services.common.errors import ServiceError, service_error_handler, generic_error_handler
from services.common.config import settings

app = FastAPI(title="adapter-service")
app.add_middleware(TraceMiddleware)
app.add_exception_handler(ServiceError, service_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

@app.get('/health')
async def health():
    return {'status':'ok','service': settings.service_name}

@app.get('/')
async def root():
    return {'service':'adapter','message':'insights adapter service skeleton'}
