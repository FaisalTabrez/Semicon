# Organizer data

Do not commit the KLA archives or extracted training arrays.

Expected extracted layout:

```text
data/raw/train/GT/*.npy
data/raw/train/NoisyLR/*.npy
data/raw/test/NoisyLR/*.npy
```

The public release inspected on 11 August 2026 contained 3,200 matching training pairs and 400 test inputs. The ZIP files also contained `__MACOSX` and `._*` metadata; repository discovery code ignores those entries.

Official download links and rules remain on the [I4C hackathon page](https://i4c.in/hackathon-2026/).

Validate an extracted training set from the repository root:

```powershell
python scripts/build_manifest.py `
  --input-dir C:\path\to\train\NoisyLR `
  --target-dir C:\path\to\train\GT `
  --dataset-root C:\path\to\train
```

The current-release count gate is 3,200 pairs. No input value is clipped or normalized during the audit.
