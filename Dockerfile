# Sử dụng Python 3.10 làm base image
FROM python:3.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cài đặt CMake trước khi cài thư viện Python
RUN apt-get update && apt-get install -y cmake

# Copy toàn bộ project vào container
COPY . /app

# Cài đặt thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Chạy ứng dụng
CMD ["python", "app.py"]
