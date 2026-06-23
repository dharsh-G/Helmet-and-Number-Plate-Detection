\# Helmet Violation Detector



A real-time helmet violation detection system using YOLOv8 and EasyOCR. Detects two-wheelers, identifies riders without helmets, and automatically reads and logs their number plates.



\---



\## Pipeline



```

Input (Video / Image / Webcam)

&#x20;       ↓

Two-Wheeler Detection (YOLOv8)

&#x20;       ↓

ROI Expansion

&#x20;       ↓

Helmet Detection (YOLOv8)

&#x20;       ↓ (no-helmet only)

Number Plate Detection (YOLOv8)

&#x20;       ↓

OCR + Plate Correction (EasyOCR)

&#x20;       ↓

violations.txt + violations.xlsx

```



\---



\## Features



\- Detects two-wheelers and identifies no-helmet violations in real time

\- Automatically crops and reads number plates using EasyOCR with preprocessing

\- Positional character correction for Indian number plates (e.g. `6→G`, `0→O` in state code)

\- Deduplication with fuzzy matching — same plate won't log twice within cooldown period

\- Voting buffer for video mode — confirms plate across multiple frames before logging

\- Saves violations to both `violations.txt` and `violations.xlsx` with timestamps

\- Supports webcam, video file, and image file inputs

\- Dark-themed Tkinter GUI with live feed, plate log panel, and source controls



\---



\## Requirements



\- Windows / Linux

\- Python 3.9+

\- NVIDIA GPU with CUDA (recommended — CPU will work but slowly)



\---



\## Installation



1\. \*\*Clone or download the project\*\*



2\. \*\*Create and activate a virtual environment\*\*

&#x20;  ```bash

&#x20;  python -m venv venv

&#x20;  venv\\Scripts\\activate        # Windows

&#x20;  source venv/bin/activate     # Linux/macOS

&#x20;  ```



3\. \*\*Install dependencies\*\*

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. \*\*Place your trained models\*\* in the `models/` folder:

&#x20;  ```

&#x20;  models/

&#x20;  ├── two\_wheeler.pt

&#x20;  ├── helmet.pt

&#x20;  └── plate.pt

&#x20;  ```



5\. \*\*Create the output folder\*\*

&#x20;  ```

&#x20;  number plate log/

&#x20;  ```



\---



\## Usage



```bash

python main.py

```



\- Click \*\*Webcam\*\* to use your live camera

\- Click \*\*Video File\*\* to select an `.mp4` / `.avi` / `.mov` file

\- Click \*\*Image File\*\* to run on a single image

\- Press \*\*▶ Start\*\* to begin detection

\- Press \*\*■ Stop\*\* to stop

\- Detected plates appear in the \*\*Plate Log\*\* panel on the right

\- All violations are saved automatically to `number plate log/violations.txt` and `number plate log/violations.xlsx`



\---



\## Project Structure



```

Helmet Detection/

├── main.py                   # Main application

├── requirements.txt

├── models/

│   ├── two\_wheeler.pt        # YOLOv8 two-wheeler detector

│   ├── helmet.pt             # YOLOv8 helmet/no-helmet classifier

│   └── plate.pt              # YOLOv8 number plate detector

└── number plate log/

&#x20;   ├── violations.txt

&#x20;   └── violations.xlsx

```



\---



\## Output Format



\*\*violations.txt\*\*

```

2025-04-25 14:33:21  |  MH 01 E 0659

2025-04-25 14:33:45  |  KA 01 W 8110

```



\*\*violations.xlsx\*\* — spreadsheet with columns: `#`, `Timestamp`, `Plate Number`



\---



\## Notes



\- GPU is strongly recommended. CPU inference will be significantly slower, especially for video.

\- OCR accuracy depends on plate visibility, angle, and image resolution. The voting buffer in video mode improves accuracy by confirming reads across multiple frames.

\- The system targets Indian number plate formats (`XX 00 XX 0000`).

