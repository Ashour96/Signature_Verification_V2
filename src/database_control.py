import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from src.database_connect import get_connection


# Connect to DataBase
conn = get_connection()

# #################################
# 1- ADD NEW USER
# #################################
def add_user(
    national_id,
    full_name,
    account_number,
    phone_number,
    email,
    account_type,
    balance,
    signature_path,
    embedding
):

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (1024,):
        raise ValueError("Embedding must contain exactly 1024 values")

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains invalid values")

    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:

                # Insert customer information
                cur.execute("""
                    INSERT INTO users (
                        nationalid,
                        fullname,
                        accountnumber,
                        phonenumber,
                        email,
                        accounttype,
                        balance,
                        signatureimagepath
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    national_id,
                    full_name,
                    account_number,
                    phone_number,
                    email,
                    account_type,
                    balance,
                    signature_path
                ))

                user_id = cur.fetchone()[0]

                # Insert signature embedding
                cur.execute("""
                    INSERT INTO signature_embeddings (
                        userid,
                        embedding
                    )
                    VALUES (%s, %s);
                """, (user_id, embedding))

        return user_id

    finally:
        conn.close()



# #################################
# 2- Retrieve user information
# #################################
def get_user(national_id):

    conn = get_connection()

    try:
        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    u.id,
                    u.nationalid,
                    u.fullname,
                    u.accountnumber,
                    u.phonenumber,
                    u.email,
                    u.accounttype,
                    u.balance,
                    u.signatureimagepath,
                    e.embedding
                FROM users u
                LEFT JOIN signature_embeddings e
                    ON u.id = e.userid
                WHERE u.nationalid = %s;
            """, (national_id,))

            result = cur.fetchone()

            if result is None:
                return None

            user = {
                "id": result[0],
                "national_id": result[1],
                "full_name": result[2],
                "account_number": result[3],
                "phone_number": result[4],
                "email": result[5],
                "account_type": result[6],
                "balance": result[7],
                "signature_path": result[8],
                "embedding": result[9]
            }

            return user

    finally:
        conn.close()



# #################################
# 3- Update user information (phone_number, email)
# #################################
def update_user(national_id, phone_number, email):

    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:

                cur.execute("""
                    UPDATE users
                    SET
                        phonenumber = %s,
                        email = %s,
                        updatedat = CURRENT_TIMESTAMP
                    WHERE nationalid = %s
                    RETURNING id;
                """, (
                    phone_number,
                    email,
                    national_id
                ))

                result = cur.fetchone()

                return result is not None

    finally:
        conn.close()



# #################################
# 4- Update the signature embedding
# #################################
def update_embedding(user_id, new_embedding):

    new_embedding = np.asarray(
        new_embedding,
        dtype=np.float32
    )

    if new_embedding.shape != (1024,):
        raise ValueError("Embedding must contain exactly 1024 values")

    if not np.isfinite(new_embedding).all():
        raise ValueError("Invalid embedding values")

    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:

                cur.execute("""
                    UPDATE signature_embeddings
                    SET embedding = %s
                    WHERE userid = %s
                    RETURNING id;
                """, (new_embedding, user_id))

                return cur.fetchone() is not None

    finally:
        conn.close()



# #################################
# 5- Delete a user (DELETE)
# #################################
def delete_user(national_id):

    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:

                cur.execute("""
                    DELETE FROM users
                    WHERE nationalid = %s
                    RETURNING id;
                """, (national_id,))

                return cur.fetchone() is not None

    finally:
        conn.close()