import pandas as pd

from typing import Literal
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

LABEL_MAP = {"Fit": 1, "Unlabeled": 0}

class InnoTripletForValidationDataset:
	def __init__(
			self,
			data_dir: str,
			split: Literal["test", "train", "triplets"] = "test",
			evaluation_mode: str = "job_to_talent"
		):
		self.data_dir = data_dir
		self.split = split
		self.evaluation_mode = evaluation_mode

		valid_evaluation_modes = {"job_to_talent", "talent_to_job"}
		if self.evaluation_mode not in valid_evaluation_modes:
			raise ValueError(f"evaluation_mode must be one of {valid_evaluation_modes}")

	def load(self) -> tuple[Dataset, dict, dict]:
		jobs_df = pd.read_csv(f"{self.data_dir}/jobs.csv")
		resumes_df = pd.read_csv(f"{self.data_dir}/resumes.csv")
		triplets_df = pd.read_csv(f"{self.data_dir}/{self.split}.csv")

		job_text = dict(zip(jobs_df["id"], jobs_df["text"].fillna("")))
		resume_text = dict(zip(resumes_df["id"], resumes_df["cleaned_text"].fillna("")))

		fit_df = triplets_df[triplets_df["label"] == "Fit"]
		if self.evaluation_mode == "job_to_talent":
			valid_queries = set(fit_df["vacancy_id"].tolist())
			triplets_df = triplets_df[triplets_df["vacancy_id"].isin(valid_queries)]
		else:  # talent_to_job
			valid_queries = set(fit_df["talent_id"].tolist())
			triplets_df = triplets_df[triplets_df["talent_id"].isin(valid_queries)]

		records = []
		for _, row in triplets_df.iterrows():
			job_id = int(row["vacancy_id"])
			talent_id = int(row["talent_id"])
			label_value = LABEL_MAP.get(row["label"], 0)

			if self.evaluation_mode == "job_to_talent":
				query_id, candidate_id = job_id, talent_id
				query, candidate = job_text.get(job_id, ""), resume_text.get(talent_id, "")
			else:
				query_id, candidate_id = talent_id, job_id
				query, candidate = resume_text.get(talent_id, ""), job_text.get(job_id, "")


			if not query or not candidate:
				print(
					f"Warning: Missing text for job_id={job_id} or talent_id={talent_id}. "
					f"Using empty string as fallback."
				)
				continue
		
			records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "label": label_value,
                }
            )

		ds = Dataset.from_pandas(pd.DataFrame(records))

		if self.evaluation_mode == "job_to_talent":
			candidates_pool = {str(k): v for k, v in resume_text.items() if v}
			print("Number of valid job queries:", len(valid_queries))
			print("Original candidate pool size (talents):", len(resume_text))
			print(f"Candidate pool size (talents): {len(candidates_pool)}")
		else:
			candidates_pool = {str(k): v for k, v in job_text.items() if v}
			print("Number of valid talent queries:", len(valid_queries))
			print("Original candidate pool size (jobs):", len(job_text))
			print(f"Candidate pool size (jobs): {len(candidates_pool)}")

		return ds, candidates_pool

class InnoTripletForTrainingDataset(TorchDataset):
	def __init__(self, data_dir: str, split: str="train"):
		self.split = split
		self.data_dir = data_dir

		jobs_df = pd.read_csv(f"{self.data_dir}/jobs.csv")
		resumes_df = pd.read_csv(f"{self.data_dir}/resumes.csv")
		triplets_df = pd.read_csv(f"{self.data_dir}/{self.split}.csv")

		job_text = dict(zip(jobs_df["id"], jobs_df["text"].fillna("")))
		resume_text = dict(zip(resumes_df["id"], resumes_df["cleaned_text"].fillna("")))

		self.records = []
		for _, row in triplets_df.iterrows():
			job_id = int(row["vacancy_id"])
			talent_id = int(row["talent_id"])
			label_value = LABEL_MAP[row["label"]]
			
			job_txt = job_text.get(job_id, "")
			resume_txt = resume_text.get(talent_id, "")
			
			if not job_txt or not resume_txt:
				continue

			self.records.append(
				{
					"query_text": job_txt,
					"candidate_text": resume_txt,
					"label": label_value,
				}
			)

		print(f"Loaded {len(self.records)} records for split {self.split}")

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx):
		return self.records[idx]