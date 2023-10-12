from elasticsearch import Elasticsearch
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ES_CONFIG = {
    "username": "elastic",
    "password": "rrP0G1zLRWyQoWXzQWgM",
    "elasticsearch_endpoint": "127.0.0.1:9200",
    "url": "https://elastic:rrP0G1zLRWyQoWXzQWgM@127.0.0.1:9200",
}
auth = (ES_CONFIG["username"], ES_CONFIG["password"])

def init_ES():
    return Elasticsearch(ES_CONFIG["url"], verify_certs=False)
