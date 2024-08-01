import requests

# Service specific config
data = {"query": "How can i query the ray StateApiClient in batch?"}
base_url = "https://ray-assistant-public-98zsh.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com"

# Requests config
path = "/stream"
full_url = f"{base_url}{path}"

resp = requests.post(full_url, json=data)

print(resp.text)

# # Constructing the new request data structure with the required 'role' field
# data = {
#     "messages": [
#         {
#             "content": "What is the default batch size for map_batches?",
#             "role": "user"  # Assuming 'user' is the correct role value. Adjust if necessary.
#         }
#     ]
# }
# # Requests config
# path = "/chat"
# full_url = f"{base_url}{path}"

# # Send POST request to the modified endpoint, including the 'role' field
# resp = requests.post(full_url, json=data)
# print(resp.text)
