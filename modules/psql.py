'''
Author: Hariz Hisham, Dec 12 2019
'''
from configparser import ConfigParser
import psycopg2


'''
Based on tutorial available at http://www.postgresqltutorial.com/postgresql-python/connect/
'''
def config(filename='auth_SJ.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
 
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
 
    return db

'''
arg should ideally be a config() file.
'''
def create_connection():
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    import psycopg2
    conn = None
    try:
        params = config()
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        
        if conn != None:
            print('Connection to database successful.')
        return conn
    except:
        logging.warning('unable to connect to database')
        exit(1)
 
    return conn
