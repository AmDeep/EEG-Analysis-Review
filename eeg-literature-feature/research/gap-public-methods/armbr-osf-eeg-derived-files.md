Fetched from https://osf.io/th2g6/ and OSF API listing https://api.osf.io/v2/nodes/th2g6/files/osfstorage/688149174ccbb4e567403929/

OSF project: ### The semi-synthetic data used
 implements the ARMBR algorithm, described in:

> **Alkhoury L**, Scanavini G, Louviot S, Radanovic A, Shah SA & Hill NJ (2025). _Artifact-Reference Multivariate Backward Regression (ARMBR): A Novel Method for EEG Blink Artifact Removal with Minimal Data Requirements._ _Journal of Neural Engineering_, 22(3). [DOI: 10.1088/1741-2552/ade566](https://doi.org/10.1088/1741-2552/ade566) [PubMed: 40527334](https://www.ncbi.nlm.nih.gov/pubmed/40527334)

The semi-synthetic data used in the ARMBR paper, are available here (under `Files`). This will allow you to test ARMBR of the semi-synthetic data used in the paper.

Full ARMBR Python and MATLAB implementations, as well as the BCI2000 GUI can be found:

[https://github.com/S-Shah-Lab/ARMBR](https://github.com/S-Shah-Lab/ARMBR)

You can also download ARMBR using:

`python -m pip install ARMBR`

If you want to install the batch files used for the BCI2000 implementation, use:

`python -m ARMBR --install-bci2000-demo`

You can then find batch files to train and apply ARMBR from BCI2000.

License: The code in this repository is released under the GNU General Public License v3.0 (`GPL-3.0-only`). See the `LICENSE.txt` file in the GitHub repository for full license terms.

Read more

## Files Preview

Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements

OSF Storage

Download As Zip

BCI2000

0 B

BCI2000 ARMBR Demo.mp4

2 Downloads

76.9 MB

Nov 4, 2025 03:54 PM

PaperTest\_SemiSyntheticData\_MNE\_Format.py

2 Downloads

514 B

Jul 23, 2025 04:40 PM

PaperTest\_SemiSyntheticData.py

3 Downloads

2.5 kB

Jul 23, 2025 04:40 PM

SemiSyntheticData

0 B

## Components

Add components to organize your projec

---

### ## Files Preview
also download ARMBR using:

`python -m pip install ARMBR`

If you want to install the batch files used for the BCI2000 implementation, use:

`python -m ARMBR --install-bci2000-demo`

You can then find batch files to train and apply ARMBR from BCI2000.

License: The code in this repository is released under the GNU General Public License v3.0 (`GPL-3.0-only`). See the `LICENSE.txt` file in the GitHub repository for full license terms.

Read more

## Files Preview

Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements

OSF Storage

Download As Zip

BCI2000

0 B

BCI2000 ARMBR Demo.mp4

2 Downloads

76.9 MB

Nov 4, 2025 03:54 PM

PaperTest\_SemiSyntheticData\_MNE\_Format.py

2 Downloads

514 B

Jul 23, 2025 04:40 PM

PaperTest\_SemiSyntheticData.py

3 Downloads

2.5 kB

Jul 23, 2025 04:40 PM

SemiSyntheticData

0 B

## Components

Add components to organize your project.

## Linked Projects

Link your project.

## Recent Activity

[Ludvik Alkhoury](https://osf.io/se5b9) updated wiki page [home](https://osf.io/project/th2g6/wiki/?wiki=npvmh) to version 8 of [Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements](https://osf.io/th2g6)

Dec 12, 2025 12:52 PM

[Ludvik Alkhoury](https://osf.io/se5b9) updated wiki page [home](https://osf.io/project/th2g6/wiki/?wiki=npvmh) to version 7 of [Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements](https://osf.io/th2g6)

Dec 12, 2025 12:52 PM

[Ludvik Alkhoury](https://osf.io/se5b9) updated wiki page [home](https://osf.io/project/th2g6/wi

OSF Sub1 folder listing proves separate clean and blink-signal MAT files (these are semi-synthetic EEG-derived data, not raw full recordings):
### Sub1_Clean_EEG.mat
developer.osf.io/#operation/nodes_files_list).

```
GET /v2/nodes/th2g6/files/osfstorage/688149174ccbb4e567403929/
```

```
HTTP 200 OK
Allow: GET, HEAD, OPTIONS
Content-Type: application/vnd.api+json ;utf-8
Vary: Accept

{
    "data": [\
        {\
            "id": "6881493e98e82bd7c7e7cd86",\
            "type": "files",\
            "attributes": {\
                "guid": "5pcyf",\
                "checkout": null,\
                "name": "Sub1_Clean_EEG.mat",\
                "kind": "file",\
                "path": "/6881493e98e82bd7c7e7cd86",\
                "size": 1780231,\
                "provider": "osfstorage",\
                "materialized_path": "/SemiSyntheticData/Sub1/Sub1_Clean_EEG.mat",\
                "last_touched": null,\
                "date_modified": "2025-07-23T20:42:38.807839Z",\
                "date_created": "2025-07-23T20:42:38.807839Z",\
                "extra": {\
                    "hashes": {\
                        "md5": "8cadc606b741d1f4da75d44d76be7762",\
                        "sha256": "0fbbe060cc55e413890e0e1e20414c2f8f96783e0cda85944cfaea771639183b"\
                    },\
                    "downloads": 2\
                },\
                "tags": [],\
                "current_user_can_comment": false,\
                "current_version": 1,\
                "show_as_unviewed": false\
            },\
            "relationships": {\
                "parent_folder": {\
                    "links": {\
                      

---

### Sub1_Blink_Signal.mat
        "html": "https://osf.io/th2g6/files/osfstorage/6881493e98e82bd7c7e7cd86",\
                "self": "https://api.osf.io/v2/files/6881493e98e82bd7c7e7cd86/",\
                "iri": "https://osf.io/5pcyf"\
            }\
        },\
        {\
            "id": "6881493e7779269db34038b1",\
            "type": "files",\
            "attributes": {\
                "guid": "qexsg",\
                "checkout": null,\
                "name": "Sub1_Blink_Signal.mat",\
                "kind": "file",\
                "path": "/6881493e7779269db34038b1",\
                "size": 597947,\
                "provider": "osfstorage",\
                "materialized_path": "/SemiSyntheticData/Sub1/Sub1_Blink_Signal.mat",\
                "last_touched": null,\
                "date_modified": "2025-07-23T20:42:38.824792Z",\
                "date_created": "2025-07-23T20:42:38.824792Z",\
                "extra": {\
                    "hashes": {\
                        "md5": "7ca8cd965cc59f3bfdf894b54e4e8ef4",\
                        "sha256": "5812db90874edd4197c0ccaac6d9968af0f23c2ec26898620b7548bb358b4d4d"\
                    },\
                    "downloads": 2\
                },\
                "tags": [],\
                "current_user_can_comment": false,\
                "current_version": 1,\
                "show_as_unviewed": false\
            },\
            "relationships": {\
                "parent_folder": {\
                    "links": {\
                 

Root listing excerpt:
### SemiSyntheticData
point](https://developer.osf.io/#operation/nodes_files_list).

```
GET /v2/nodes/th2g6/files/osfstorage/
```

```
HTTP 200 OK
Allow: GET, HEAD, OPTIONS
Content-Type: application/vnd.api+json ;utf-8
Vary: Accept

{
    "data": [\
        {\
            "id": "688148bd60f211d8ed1f6475",\
            "type": "files",\
            "attributes": {\
                "guid": "mce29",\
                "checkout": null,\
                "name": "PaperTest_SemiSyntheticData_MNE_Format.py",\
                "kind": "file",\
                "path": "/688148bd60f211d8ed1f6475",\
                "size": 514,\
                "provider": "osfstorage",\
                "materialized_path": "/PaperTest_SemiSyntheticData_MNE_Format.py",\
                "last_touched": null,\
                "date_modified": "2025-07-23T20:40:29.471858Z",\
                "date_created": "2025-07-23T20:40:29.471858Z",\
                "extra": {\
                    "hashes": {\
                        "md5": "87014c65c4be94ac1b6723b1f6659b54",\
                        "sha256": "fb5826428c70bccdc2e3c0ea5298b1afed858de307d36faf16ab826bf41a7867"\
                    },