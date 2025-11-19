import unittest
import subprocess

def dataset_dict():
    datasets = {}
    dataset_arb = [
        "coauth-DBLP",
        "coauth-MAG-Geology",
        "email-Enron",
        "tags-math-sx",
        "contact-high-school",
        "contact-primary-school",
        "NDC-substances"
    ]
    datasets_CHLP = [
        "IMDB",
        "COURSERA",
        "ARXIV"
    ]
    negative_methods = [
        "SizedHypergraphNegativeSampler",
        "MotifHypergraphNegativeSampler",
        "CliqueHypergraphNegativeSampler"
    ]
    hlp_methods = ["CommonNeighbors"] 

    ns_hlp_union = []
    for ns in negative_methods:
        for hlp in hlp_methods:
            ns_hlp_union.append([ns, hlp])
    
    for dataset in dataset_arb:
        datasets[dataset] = {"methods": ns_hlp_union}
        
    for dataset in datasets_CHLP:
        datasets[dataset] = {"methods": ns_hlp_union}

    return datasets

def create_pipelines_comand():
    datasets = dataset_dict()
    pipelines = []

    for dataset_name, content in datasets.items():
        for ns, hlp in content["methods"]:
            cmd = (
                f"uv run pipeline "
                f"--dataset_name {dataset_name} "
                f"--hlp_method {hlp} "
                f"--negative_sampling {ns}"
                f" --test True"
            )
            pipelines.append(cmd)
    
    return pipelines

class TestPipelineExecution(unittest.TestCase):
    
    def test_pipeline_execution(self):
        pipelines = create_pipelines_comand()
        for cmd in pipelines:
            with self.subTest(cmd=cmd):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode == 0 and result.stdout.strip() != "":
                    print(f"[OK] {cmd}")
                elif result.returncode != 0:
                    print(f"[FAIL] {cmd} (exit {result.returncode})")
                    print(f"STDERR:\n{result.stderr}")
                else:
                    print(f"[FAIL] {cmd} — Nessun output prodotto")

                self.assertEqual(
                    result.returncode, 0,
                    msg=f"Pipeline fallita (exit {result.returncode}): {cmd}\nSTDERR:\n{result.stderr}"
                )
                
                self.assertTrue(
                    result.stdout.strip() != "",
                    msg=f"Pipeline terminata senza output: {cmd}"
                )
                
if __name__ == "__main__":
     unittest.main()