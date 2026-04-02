import os

path = r'C:\Users\user\Desktop\plantdisease-ml-assignment-summative\PlantDisease-Analysis\frontend\src\App.jsx'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('LeafScope Platform', 'SYSTEM DIAGNOSTICS // V1.0')
content = content.replace('API: {API_BASE_URL}', 'CONNECTION: SECURE')
content = content.replace('API {apiHealthy ? "Healthy" : "Unavailable"}', 'SYSTEM {apiHealthy ? "ONLINE" : "OFFLINE"}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
