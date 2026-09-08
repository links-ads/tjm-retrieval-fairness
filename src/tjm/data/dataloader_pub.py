import pandas as pd

from typing import Literal
from datasets import Dataset

LABEL_MAP = {"Fit": 1, "Unlabeled": 0}

class InnoTripletWithPublicationForValidationDataset:
	def __init__(
			self,
			data_dir: str,
			split: Literal["train", "test", "triplets"] = "test",
			evaluation_mode: Literal["job_to_talent", "talent_to_job"] = "job_to_talent",
			include_publications: bool = True,
			publication_sequence: Literal[None, "additional", "first", "last"] | list[Literal["additional", "first", "last"]] = None,
			publication_selection_mode: Literal["random", "by_date"] = "random",
			publication_text_fields: Literal["title", "abstract", "both"] = "both",
			max_publications: int = 5,
			random_seed: int = 42,
		):
		self.class_name = 'InnoTripletWithPublicationForValidationDataset'
		self.data_dir = data_dir
		self.split = split
		self.evaluation_mode = evaluation_mode
		self.publication_selection_mode = publication_selection_mode
		self.publication_sequence = publication_sequence
		self.publication_text_fields = publication_text_fields
		self.include_publications = include_publications
		self.max_publications = max_publications
		self.random_seed = random_seed

		if self.max_publications < 1:
			raise ValueError("max_publications must be >= 1")

		self._check_valid_evaluation_mode()
		self._check_valid_publication_sequence()
		self._check_valid_publication_selection_mode()
		self._check_valid_publication_text_fields()

		print(
			f"Initialized {self.class_name} with evaluation_mode={self.evaluation_mode}, "
			f"publication_selection_mode={self.publication_selection_mode}, "
			f"publication_sequence={self.publication_sequence}, "
			f"publication_text_fields={self.publication_text_fields}, "
			f"include_publications={self.include_publications}, "
			f"max_publications={self.max_publications}"
		)

	def _check_valid_evaluation_mode(self):
		valid_modes = {"job_to_talent", "talent_to_job"}
		if self.evaluation_mode not in valid_modes:
			raise ValueError(f"evaluation_mode must be one of {valid_modes}")
	
	def _check_valid_publication_selection_mode(self):
		valid_modes = {"random", "by_date"}
		if self.publication_selection_mode not in valid_modes:
			raise ValueError(f"publication_selection_mode must be one of {valid_modes}")
	
	def _check_valid_publication_text_fields(self):
		valid_fields = {"title", "abstract", "both"}
		if self.publication_text_fields not in valid_fields:
			raise ValueError(f"publication_text_fields must be one of {valid_fields}")
		
	def _check_valid_publication_sequence(self):
		valid_sequence_modes = {"additional", "first", "last"}
		if self.publication_sequence is None:
			self.publication_sequences = None
		elif isinstance(self.publication_sequence, str):
			if self.publication_sequence not in valid_sequence_modes:
				raise ValueError(
					f"publication_sequence must be None, one of {valid_sequence_modes}, "
					f"or a list of those values"
				)
			self.publication_sequences = {self.publication_sequence}
		elif isinstance(self.publication_sequence, list):
			seq_set = {str(s) for s in self.publication_sequence}
			if not seq_set.issubset(valid_sequence_modes):
				raise ValueError(
					f"publication_sequence list values must be in {valid_sequence_modes}"
				)
			self.publication_sequences = seq_set if seq_set else None
		else:
			raise ValueError(
				"publication_sequence must be None, a sequence string, or a list of sequence strings"
			)

	def _select_publications(self, talent_publications: pd.DataFrame) -> pd.DataFrame:
		if talent_publications.empty:
			print("No publications found for this talent.")
			return talent_publications

		# Optional sequence pre-filter: None means no sequence filtering.
		if self.publication_sequences is not None:
			talent_publications = talent_publications[
				talent_publications["Sequence"].fillna("").str.lower().isin(self.publication_sequences)
			]

		if talent_publications.empty:
			print("No publications left after sequence filtering.")
			return talent_publications

		if self.publication_selection_mode == "random":
			n = min(self.max_publications, len(talent_publications))
			return talent_publications.sample(
				n=n,
				random_state=self.random_seed,
				replace=False,
			)

		if self.publication_selection_mode == "by_date":
			pubs = talent_publications.copy()
			pubs["PublicationDateParsed"] = pd.to_datetime(
				pubs["PublicationDate"], errors="coerce"
			)
			return pubs.sort_values(
				"PublicationDateParsed", ascending=False, na_position="last"
			).head(self.max_publications)

		# Fallback for safety; should not happen because mode is validated.
		return talent_publications.head(self.max_publications)

	def _build_publication_text(self, selected_publications: pd.DataFrame) -> str:
		parts = []
		for _, pub in selected_publications.iterrows():
			title = str(pub.get("Title", "") or "").strip()
			abstract = str(pub.get("Abstract", "") or "").strip()

			if self.publication_text_fields == "title":
				content = f"Title: {title}"
			elif self.publication_text_fields == "abstract":
				content = f"Abstract: {abstract}"
			else:
				content = f"Title: {title}\nAbstract: {abstract}".strip()

			if content:
				parts.append(content)

		if not parts:
			print("No valid publication text found after building. Returning empty string.")
			return ""

		return "\n\n".join(parts)

	def load(self) -> tuple[Dataset, dict]:
		jobs_df = pd.read_csv(f"{self.data_dir}/jobs.csv")
		resumes_df = pd.read_csv(f"{self.data_dir}/resumes.csv")
		triplets_df = pd.read_csv(f"{self.data_dir}/{self.split}.csv")
		publications_df = pd.read_csv(f"{self.data_dir}/publications.csv")

		if self.split != "triplets":
			split_job_ids = set(pd.to_numeric(triplets_df["vacancy_id"], errors="coerce").dropna().astype(int))
			split_talent_ids = set(pd.to_numeric(triplets_df["talent_id"], errors="coerce").dropna().astype(int))
			jobs_df = jobs_df[pd.to_numeric(jobs_df["id"], errors="coerce").isin(split_job_ids)].copy()
			resumes_df = resumes_df[pd.to_numeric(resumes_df["id"], errors="coerce").isin(split_talent_ids)].copy()

		job_text = dict(zip(jobs_df["id"], jobs_df["text"].fillna("")))
		resume_text = dict(zip(resumes_df["id"], resumes_df["cleaned_text"].fillna("")))

		fit_triplets = triplets_df[triplets_df["label"] == "Fit"].copy()
		fit_talents = fit_triplets["talent_id"].dropna().astype(int).unique()

		pub_counts = publications_df.groupby("TalentId")["PublicationId"].nunique()
		fit_talent_pub_counts = pub_counts.reindex(fit_talents).dropna()
		fit_talents_with_publications = set(
			fit_talent_pub_counts[fit_talent_pub_counts > 0].index.astype(int).tolist()
		)
		fit_triplets_with_published_talents = fit_triplets[
			fit_triplets["talent_id"].astype(int).isin(fit_talents_with_publications)
		]

		# Build resume text enriched with selected publication snippets per talent.
		resume_text_with_publications = dict(resume_text)
		if self.include_publications:
			pubs_by_talent = {
				int(talent_id): df
				for talent_id, df in publications_df.groupby("TalentId")
			}
			for talent_id, talent_publications in pubs_by_talent.items():
				base_resume = resume_text.get(talent_id, "")
				if talent_id not in resume_text:
					continue

				selected_publications = self._select_publications(talent_publications)
				publication_text = self._build_publication_text(selected_publications)

				if base_resume and publication_text:
					resume_text_with_publications[talent_id] = (
						f"Resume:{base_resume}\n\nPublications:\n{publication_text}"
					)
				elif publication_text:
					resume_text_with_publications[talent_id] = f"Publications:\n{publication_text}"

		if self.evaluation_mode == "talent_to_job":
			valid_queries = fit_talents_with_publications
			filtered_triplets = triplets_df[triplets_df["talent_id"].isin(valid_queries)]
			print("Number of valid talent queries with publications:", len(valid_queries))
		else:  # job_to_talent
			valid_queries = set(
				fit_triplets_with_published_talents["vacancy_id"].dropna().astype(int).tolist()
			)
			filtered_triplets = triplets_df[triplets_df["vacancy_id"].isin(valid_queries)]
			print("Number of valid job queries with published talents:", len(valid_queries))
			print("")

		records = []
		for _, row in filtered_triplets.iterrows():
			job_id = int(row["vacancy_id"])
			talent_id = int(row["talent_id"])
			label_value = LABEL_MAP.get(row["label"], 0)

			if self.evaluation_mode == "job_to_talent":
				query_id, candidate_id = job_id, talent_id
				query, candidate = (
					job_text.get(job_id, ""),
					resume_text_with_publications.get(talent_id, resume_text.get(talent_id, "")),
				)
			else:
				query_id, candidate_id = talent_id, job_id
				query, candidate = (
					resume_text_with_publications.get(talent_id, resume_text.get(talent_id, "")),
					job_text.get(job_id, ""),
				)

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
			candidates_pool = {
				str(talent_id): resume_text_with_publications.get(talent_id, cv_text)
				for talent_id, cv_text in resume_text.items()
				if resume_text_with_publications.get(talent_id, cv_text)
			}
			print("Original candidate pool size (talents):", len(resume_text))
			print(f"Candidate pool size (talents): {len(candidates_pool)}")
		else:
			candidates_pool = {str(k): v for k, v in job_text.items() if v}
			print("Original candidate pool size (jobs):", len(job_text))
			print(f"Candidate pool size (jobs): {len(candidates_pool)}")

		return ds, candidates_pool