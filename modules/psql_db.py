'''
postgress db

by: Dan Trepanier

Sep 14, 2019
'''
import logging
import psycopg2

ACCOUNT = {'host':'sanjose','database':'atlas','user':'student'}

def connect():
    try:
        conn = psycopg2.connect("host=%s dbname=%s user=%s " % (ACCOUNT['host'],ACCOUNT['database'], ACCOUNT['user']))
    except:
        logging.warning('unable to connect to: %s' % ACCOUNT)
        exit(1)
    return conn

def query(sql_stmt, columns):
    answer = []
    try:
        conn = connect()
        cursor= conn.cursor()
        cursor.execute(sql_stmt)
        count = cursor.rowcount
        for i in range(count):
            x = cursor.fetchone()
            if len(x) != len(columns):
                for h,r in zip(columns, x):
                    print(h,r)
                assert len(x) == len(columns),'x contains %d items vs. columns with %d items\ncolumns: %s\nx   : %s' % (len(x), len(columns), str(columns), str(x))
            answer += [dict(zip(columns,x))]
    except:
        logging.warning( 'error while running: %s' % sql_stmt)
    
    return answer