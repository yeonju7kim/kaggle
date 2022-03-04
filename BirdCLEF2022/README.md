## Files 설명

1. train_metadata.csv

- primary_label : bird species
- secondary_labels: Background species
- author 
- filename
- rating: quality rating
  - 1~5, 0은 평가가 없을 때

2. train_audio/ - downsampled to 32 kHz

3. testsoundscapes/ - 1개의 test data만 공개

4. test.csv 
- row_id
- file_id 
- bird 
- end_time - The last second of the 5 second time window (5, 10, 15, etc).

5. sample_submission.csv
- row_id
- target

6. scored_birds.json - scored된 종 리스트

7. eBirdTaxonomyv2021.csv - 다른 종 간의 relationship

## Code 설명
- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named submission.csv
