"""Honor the existing PORT environment variable without shell interpolation."""
import os, uvicorn
if __name__=='__main__':
    port=int(os.getenv('PORT','8001'))
    if not 1<=port<=65535:raise ValueError('Invalid PORT')
    uvicorn.run('production_v31.api:app',host='0.0.0.0',port=port)
