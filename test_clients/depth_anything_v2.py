import requests
import base64
import io
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import time

def test_depth_api(image_path, api_url="http://134.245.232.230:8001", depth_type="relative", 
                  max_depth=None, model_size="small", optimization="basic"):
    """
    Test the Depth Anything API with a local image
    
    Args:
        image_path: Path to the input image
        api_url: Base URL of the API
        depth_type: Type of depth estimation ('relative' or 'metric')
        max_depth: Maximum depth value for metric depth (20 for indoor, 80 for outdoor)
        model_size: Size of the model ('small', 'base', or 'large')
        optimization: Level of optimization ('none', 'basic', or 'max')
    """
    # Check API health
    try:
        health_response = requests.get(f"{api_url}/health")
        if health_response.status_code != 200:
            print(f"API health check failed: {health_response.text}")
            return
        print("API health check passed")
        print(f"Loaded models: {health_response.json().get('loaded_models', [])}")
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to API at {api_url}. Is the server running?")
        return
    
    # Build API parameters
    params = {
        "model_size": model_size,
        "depth_type": depth_type,
        "optimization": optimization
    }
    
    if depth_type == "metric" and max_depth is not None:
        params["max_depth"] = max_depth
    
    # Test JSON response endpoint
    try:
        print(f"Sending request with parameters: {params}")
        start_time = time.time()
        
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{api_url}/predict",
                params=params,
                files={'file': f}
            )
        
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.3f} seconds")
        
        if response.status_code != 200:
            print(f"Error from API: {response.text}")
            return
        
        result = response.json()
        if not result['success']:
            print(f"API returned error: {result.get('error', 'Unknown error')}")
            return
        
        # Print depth information
        print(f"Depth type: {result.get('depth_type', 'relative')}")
        print(f"Model size: {result.get('model_size', 'small')}")
        if result.get('max_depth'):
            print(f"Max depth setting: {result.get('max_depth')}")
        print(f"Actual depth range: {result.get('min_depth_value', 'N/A')} to {result.get('max_depth_value', 'N/A')}")
        print(f"Server-side inference time: {result.get('inference_time', 'N/A'):.3f} seconds")
        print(f"Network overhead: {total_time - result.get('inference_time', 0):.3f} seconds")
        
        # Decode the base64 image
        depth_image_data = base64.b64decode(result['depth_image'])
        depth_image = Image.open(io.BytesIO(depth_image_data))
        
        # Save the depth map
        output_path = f"depth_output_{depth_type}_{model_size}_{optimization}.png"
        depth_image.save(output_path)
        print(f"Depth map saved to {output_path}")
        
        # Display the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        original_img = Image.open(image_path)
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Choose appropriate colormap for depth visualization
        cmap = 'plasma'
        
        # Depth map
        axes[1].imshow(depth_image, cmap=cmap)
        title = f"Depth Map ({depth_type}"
        if depth_type == "metric":
            title += f", max={max_depth}"
        title += f", {model_size}, opt={optimization})"
        axes[1].set_title(title)
        axes[1].axis("off")
        
        plt.tight_layout()
        comparison_filename = f"comparison_{depth_type}_{model_size}_{optimization}.png"
        plt.savefig(comparison_filename)
        print(f"Comparison image saved to {comparison_filename}")
        
        # Optionally display the plot
        # plt.show()
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")

def benchmark_api(image_path, api_url="http://localhost:8000", iterations=3):
    """Run benchmarks with different configurations"""
    print("\n" + "="*50)
    print("BENCHMARKING DEPTH ANYTHING API")
    print("="*50)
    
    # Define configurations to test
    configs = [
        {"model_size": "small", "depth_type": "relative", "optimization": "none"},
        {"model_size": "small", "depth_type": "relative", "optimization": "basic"},
        {"model_size": "small", "depth_type": "relative", "optimization": "max"},
        {"model_size": "small", "depth_type": "metric", "max_depth": 80, "optimization": "basic"},
        {"model_size": "base", "depth_type": "relative", "optimization": "basic"},
        # Uncomment if you want to test large model (takes much longer)
        # {"model_size": "large", "depth_type": "relative", "optimization": "basic"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting configuration: {config}")
        config_times = []
        
        # First request will be slower due to model loading
        print("Warming up (first request)...")
        warmup = test_depth_api(image_path, api_url, **config)
        if warmup and warmup.get('inference_time'):
            print(f"Warmup time: {warmup.get('inference_time'):.3f}s")
        
        # Run benchmark iterations
        print(f"Running {iterations} iterations...")
        for i in range(iterations):
            start = time.time()
            result = test_depth_api(image_path, api_url, **config)
            end = time.time()
            
            if result and result.get('inference_time'):
                config_times.append(result.get('inference_time'))
                print(f"  Iteration {i+1}/{iterations}: {result.get('inference_time'):.3f}s")
        
        if config_times:
            avg_time = sum(config_times) / len(config_times)
            results.append({
                "config": config,
                "avg_time": avg_time,
                "min_time": min(config_times),
                "max_time": max(config_times)
            })
            print(f"Average time: {avg_time:.3f}s")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    # Sort by average time
    results.sort(key=lambda x: x["avg_time"])
    
    for i, r in enumerate(results):
        config_str = ", ".join([f"{k}={v}" for k, v in r["config"].items()])
        print(f"{i+1}. {config_str}")
        print(f"   Average: {r['avg_time']:.3f}s, Min: {r['min_time']:.3f}s, Max: {r['max_time']:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Depth Anything V2 API")
    parser.add_argument("--image_path", help="Path to the input image", default="images/spot_image_back_fisheye_image_1744204082.jpg")
    parser.add_argument("--api-url", default="http://134.245.232.230:8001", help="Base URL of the API")
    parser.add_argument("--depth-type", choices=["relative", "metric"], default="relative", 
                        help="Type of depth estimation")
    parser.add_argument("--max-depth", type=float, default=5, 
                        help="Maximum depth value for metric depth (20 for indoor, 80 for outdoor)")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small",
                        help="Size of the model to use")
    parser.add_argument("--optimization", choices=["none", "basic", "max"], default="basic",
                        help="Level of optimization to apply")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark with different configurations")
    parser.add_argument("--benchmark-iterations", type=int, default=3, 
                        help="Number of iterations for each configuration in benchmark mode")
    
    args = parser.parse_args()
    
    # If using metric depth but no max_depth specified, use defaults based on a simple prompt
    if args.depth_type == "metric" and args.max_depth is None:
        scene_type = input("Is this an indoor or outdoor scene? (indoor/outdoor): ").lower().strip()
        if scene_type == "indoor":
            args.max_depth = 20.0
        else:
            args.max_depth = 80.0
    
    if args.benchmark:
        benchmark_api(args.image_path, args.api_url, args.benchmark_iterations)
    else:
        test_depth_api(
            args.image_path, 
            api_url=args.api_url, 
            depth_type=args.depth_type,
            max_depth=args.max_depth,
            model_size=args.model_size,
            optimization=args.optimization
        ) 