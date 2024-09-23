import boto3
import logging
from botocore.exceptions import ClientError


def upload_file_to_s3(file_path, s3_folder, s3_file_name):
    """
    Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_path is used
    """
    bucket = 'data-visualization-benchmark'
    # If S3 object_name was not specified, use file_path 
    object_name = f'{s3_folder}/{s3_file_name}'

    # Upload the file
    s3_client = boto3.client('s3', region_name='us-west-2')

    try:
        response = s3_client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

if __name__ == "__main__":
    # Example usage
    file_path = '/Users/arnav/Desktop/contextvis/vlm-datavis-benchmark/analysis/tacl_analysis/responses/incontext_instructions_question/chartqa-test-categorical_responses.csv'
    s3_folder = 'mini-vlat'
    s3_file_name = 'response/model_response/chartqa-test-categorical_responses.csv'
    # Upload the file
    success = upload_file_to_s3(file_path, s3_folder=s3_folder, s3_file_name=s3_file_name)
    if success:
        print("File uploaded successfully.")
    else:
        print("Upload failed.")
