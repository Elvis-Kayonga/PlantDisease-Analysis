"""
Locust load testing script for Plant Disease Detection API.
Tests prediction endpoint with multiple concurrent requests.
"""

from locust import HttpUser, task, between, events
import random
import time
import json
from pathlib import Path

# Test image paths (should exist in data/test)
TEST_IMAGE_DIR = "data/test"
TEST_IMAGES = []


def setup_test_images():
    """Load available test images."""
    global TEST_IMAGES
    test_dir = Path(TEST_IMAGE_DIR)
    if test_dir.exists():
        TEST_IMAGES = list(test_dir.glob("*/*.jpg")) + list(test_dir.glob("*/*.JPG"))
    
    if not TEST_IMAGES:
        # Create dummy test images if none found
        print(f"Warning: No test images found in {TEST_IMAGE_DIR}")
        print("Test will use placeholder image data")


class PlantDiseaseAPIUser(HttpUser):
    """Locust user for testing Plant Disease Detection API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when user starts."""
        self.client.verify = False  # Disable SSL verification for local testing
        
        # Check API health
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("API is not healthy")
    
    @task(1)
    def health_check(self):
        """Task: Check API health."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def get_model_status(self):
        """Task: Get model status."""
        with self.client.get(
            "/model-status",
            catch_response=True,
            name="/model-status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def predict_disease(self):
        """Task: Predict disease from image."""
        try:
            # Select random test image or create test data
            if TEST_IMAGES:
                image_path = random.choice(TEST_IMAGES)
                with open(image_path, 'rb') as f:
                    files = {'file': f}
                    response = self.client.post(
                        "/predict",
                        files=files,
                        catch_response=True,
                        name="/predict"
                    )
            else:
                # Create minimal test image data (1x1 pixel)
                import io
                from PIL import Image
                
                img = Image.new('RGB', (128, 128), color='red')
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
                response = self.client.post(
                    "/predict",
                    files=files,
                    catch_response=True,
                    name="/predict"
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    response.success()
                else:
                    response.failure("API returned success=false")
            else:
                response.failure(f"Status code: {response.status_code}")
        
        except Exception as e:
            self.client.get(
                "/",
                catch_response=True,
                name="/predict"
            ).failure(f"Exception: {str(e)}")
    
    @task(2)
    def get_metrics(self):
        """Task: Get model metrics."""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Event handlers for reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("\n" + "="*70)
    print("PLANT DISEASE DETECTION API - LOAD TEST")
    print("="*70)
    print(f"Target: {environment.host}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "="*70)
    print("LOAD TEST RESULTS SUMMARY")
    print("="*70)
    
    stats = environment.stats
    
    # Request summary
    print("\nRequest Statistics:")
    print("-" * 70)
    print(f"{'Endpoint':<20} {'Requests':<15} {'Failures':<15} {'Median (ms)':<15}")
    print("-" * 70)
    
    for key in sorted(stats.entries.keys()):
        entry = stats.entries[key]
        median_ms = entry.response_times.median()
        print(
            f"{entry.name:<20} {entry.num_requests:<15} "
            f"{entry.num_failures:<15} {median_ms:<15.2f}"
        )
    
    print("\n" + "-" * 70)
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Failed by: {stats.total.num_failures}/{stats.total.num_requests}")
    
    # Response time stats
    if stats.total.num_requests > 0:
        print(f"\nResponse Time Statistics:")
        print(f"  Min: {stats.total.min_response_time:.2f} ms")
        print(f"  Max: {stats.total.max_response_time:.2f} ms")
        print(f"  Median: {stats.total.median_response_time:.2f} ms")
        print(f"  Average: {stats.total.avg_response_time:.2f} ms")
    
    print("\nEnd time: " + time.strftime('%Y-%m-%d %H:%M:%S'))
    print("="*70 + "\n")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Called when locust is quitting."""
    print("\nTest execution completed. Cleaning up...")


if __name__ == "__main__":
    setup_test_images()
    print("Locust script configured. Use 'locust -f locustfile.py' to run tests.")
