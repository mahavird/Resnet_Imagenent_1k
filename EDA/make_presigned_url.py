# make_presigned_multipart.py
import os, math, json
import boto3

BUCKET = "imagenetdataresnet"
s3 = boto3.client("s3")

def start_multipart(key):
    resp = s3.create_multipart_upload(Bucket=BUCKET, Key=key)
    return resp["UploadId"]

def presign_parts(key, upload_id, num_parts, expires_hours=12):
    out = []
    for pn in range(1, num_parts + 1):
        url = s3.generate_presigned_url(
            "upload_part",
            Params={"Bucket": BUCKET, "Key": key, "UploadId": upload_id, "PartNumber": pn},
            ExpiresIn=expires_hours * 3600
        )
        out.append({"PartNumber": pn, "Url": url})
    return out

def plan_parts(file_size_bytes, part_size_bytes=100*1024*1024):  # 100MB parts
    return math.ceil(file_size_bytes / part_size_bytes), part_size_bytes

def main(local_path, s3_key, json_out):
    # if you don't have the file locally, just ask your collaborator for its size in bytes
    size = os.path.getsize(local_path) if os.path.exists(local_path) else None
    if size is None:
        raise SystemExit("File not found locally. Provide a path that exists OR modify the script to pass a known size.")
    num_parts, part_size = plan_parts(size)
    upload_id = start_multipart(s3_key)
    parts = presign_parts(s3_key, upload_id, num_parts)
    payload = {
        "Bucket": BUCKET,
        "Key": s3_key,
        "UploadId": upload_id,
        "PartSize": part_size,
        "NumParts": num_parts,
        "Parts": parts
    }
    with open(json_out, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {json_out} with {num_parts} part URLs.\nUploadId={upload_id}")

if __name__ == "__main__":
    # Example usage (edit local paths/S3 keys to taste)
    main("/tmp/ILSVRC2012_img_train.tar", "uploads/ILSVRC2012_img_train.tar", "train_links.json")
    main("/tmp/ILSVRC2012_img_val.tar",   "uploads/ILSVRC2012_img_val.tar",   "val_links.json")
