import boto3

s3 = boto3.client('s3')
url = s3.generate_presigned_url(
    ClientMethod='put_object',
    Params={'Bucket': 'data-bucket-imagenet', 'Key': 'uploads/myfile.txt'},
    ExpiresIn=3600  # link valid for 1 hour
)
print(url)