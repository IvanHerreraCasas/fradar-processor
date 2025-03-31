# core/utils/secrets.py

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_ftp_password(server_alias: str) -> Optional[str]:
    """
    Retrieves the FTP password for a given server alias from environment variables.
    Constructs the environment variable name as 'FTP_PASSWORD_<ALIAS_UPPERCASE>'.
    """
    if not server_alias:
        logger.error("Cannot retrieve FTP password: server_alias is missing.")
        return None
    try:
        env_var_suffix = server_alias.upper().replace('-', '_').replace(' ', '_')
        env_var_name = f"FTP_PASSWORD_{env_var_suffix}"
        password = os.environ.get(env_var_name)
        if password is not None:
            logger.debug(f"Retrieved password for alias '{server_alias}' from environment variable.") # DO NOT log name/value
            return password
        else:
            logger.warning(f"FTP password environment variable (like '{env_var_name}') is not set for server alias '{server_alias}'.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving FTP password for alias '{server_alias}': {e}", exc_info=True)
        return None