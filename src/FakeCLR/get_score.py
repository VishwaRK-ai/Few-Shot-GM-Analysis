import os
import zipfile
import torch_fidelity

def main():
    print("1/2: Preparing real dataset...")
    os.makedirs('real_temp', exist_ok=True)
    with zipfile.ZipFile('datasets/obama.zip', 'r') as zip_ref:
        zip_ref.extractall('real_temp')

    print("2/2: Calculating FID Score...")
    metrics = torch_fidelity.calculate_metrics(
        input1='fid_temp_fake',
        input2='real_temp',
        cuda=False,
        isc=False,
        fid=True,
        datasets_num_workers=0,
        samples_find_deep=True
    )

    fid_score = metrics['frechet_inception_distance']
    
    # Format the result
    result_text = f"FINAL FID SCORE: {fid_score:.4f}\n"

    print("\n" + "="*40)
    print(result_text.strip())
    print("="*40 + "\n")

    # Write to a file
    with open("fid_result.txt", "w") as f:
        f.write("Evaluation Metric: Frechet Inception Distance (FID-100 Proxy)\n")
        f.write("Model: FakeCLR (340 kimg)\n")
        f.write("-" * 40 + "\n")
        f.write(result_text)
    
    print("Successfully saved to fid_result.txt!")

if __name__ == '__main__':
    main()