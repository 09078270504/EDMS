# Handles data extraction, metadata saving, and document status updates
import logging
from ..models import Documents, Document_Metadata

logger = logging.getLogger(__name__)

class Ml_Database_Service:
    def save_ml_metadata(document_id, extracted_data):
        try:
            document = Documents.objects.get(id=document_id)
            logger.info(f"Saving metadata for '{document.document_name}' (ID {document_id})")

            # Extract financial and entity-related metadata
            financial_data = extracted_data.get('financial', {})
            entities_data = extracted_data.get('entities', {})

            # create or update the metadata
            metadata, created = Document_Metadata.objects.update_or_create(
                document=document,
                defaults={
                    'amount': financial_data.get('amount'),
                    'currency': financial_data.get('currency', 'PHP'),
                    'date_issued': financial_data.get('date_issued'),
                    'client_name': entities_data.get('client_name', ''),
                    'vendor_name': entities_data.get('vendor_name', ''),
                    'invoice_number': entities_data.get('invoice_number', ''),
                    'additional_data': extracted_data # It stores all data in JSON form
                }
            )

            if created:
                logger.info(f"Metadata created for document ID {document_id}")
            else:
                logger.info(f"Metadata updated for document ID {document_id}")


            # update document status
            document.status = 'ready'
            document.save()
            logger.info(f"Document {document_id} status updated to 'ready' ")

            return metadata

        except Documents.DoesNotExist:
            # Handle case where document ID is invalid
            error_msg = f"Document with ID {document_id} is not found."
            logger.error(error_msg)
            raise
        
        except Exception as e:
            # Catch and log any other unexpected errors
            logger.error(f"Failed to save metadata for document {document_id}: {str(e)}")
            raise