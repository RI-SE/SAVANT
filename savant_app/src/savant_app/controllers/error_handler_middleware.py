from ..services.exceptions import DomainException, InternalException
import logging
import functools

logger = logging.getLogger(__name__)

def error_handler(func):
    """
    Middleware to handle errors in controllers.
    
    This includes converting backend errors and
    raising frontend friendly errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DomainException as e:
            logger.warning(
                f"""
                Domain exception of type {type(e)} occured.
                Exception info: {str(e)}
                custom log message: {e.log_message}
                """, 
                exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"""
                Unexpected error of type {type(e)} occured.
                Exception info: {str(e)}
                """, 
                exc_info=True)
            raise InternalException(
                """
                An unexpected error occurred.\nPlease contact support.\nDetails logged.
                """) from e

    return wrapper 