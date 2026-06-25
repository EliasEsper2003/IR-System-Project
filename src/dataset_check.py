from services.database_service import get_document

doc_id = input("Enter Doc ID: ")

document = get_document(doc_id)

print(document[:500])