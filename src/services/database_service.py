import os
import sqlite3


def get_database_path(
    dataset_name
):

    folder_name = (
        dataset_name
        .replace("/", "_")
        .replace("-", "_")
    )

    os.makedirs(
        "databases",
        exist_ok=True
    )

    return (
        f"databases/{folder_name}.db"
    )


def create_database(
    dataset_name
):

    db_path = get_database_path(
        dataset_name
    )

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            content TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_documents(
    dataset_name,
    doc_ids,
    raw_documents
):

    db_path = get_database_path(
        dataset_name
    )

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    data = list(
        zip(
            doc_ids,
            raw_documents
        )
    )

    cursor.executemany(
        """
        INSERT OR REPLACE INTO documents
        (doc_id, content)
        VALUES (?, ?)
        """,
        data
    )

    conn.commit()
    conn.close()

    print(
        "Documents Saved To Database"
    )


def database_has_documents(
    dataset_name
):

    db_path = get_database_path(
        dataset_name
    )

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM documents"
    )

    count = cursor.fetchone()[0]

    conn.close()

    return count > 0


def get_document(
    dataset_name,
    doc_id
):

    db_path = get_database_path(
        dataset_name
    )

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT content
        FROM documents
        WHERE doc_id = ?
        """,
        (doc_id,)
    )

    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0]

    return None


def get_first_document(
    dataset_name
):

    db_path = get_database_path(
        dataset_name
    )

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT doc_id, content
        FROM documents
        LIMIT 1
        """
    )

    row = cursor.fetchone()

    conn.close()

    return row