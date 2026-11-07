import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np


def get_connection():

    conn = psycopg2.connect(
        host="localhost",
        database="signature_verification",
        user="postgres",
        password="Ashour_21032026",
        port=5432
    )

    # Register pgvector
    register_vector(conn)

    return conn