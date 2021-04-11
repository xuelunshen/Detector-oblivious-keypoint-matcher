# Branch 🌿 SP-SP-RAND

|                                    Experiments                                    	|  p   |  d   |  c   |
| :--------------------------------------------------------------------------------------:	| :--: | :--: | :--: |
| [main](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/main)		|  --  |  --  |  --  |
| [#1](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-SP-SP)		|  SP  |  SP  |  SP  |
| [#2](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/R2D2-SP-SP)	| R2D2 |  SP  |  SP  |
| [#3](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-R2D2-SP)	|  SP  | R2D2 |  SP  |
| [#4](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-SP-R2D2)	|  SP  |  SP  | R2D2 |
| 👉[#5](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-SP-RAND)	|  SP  |  SP  | RAND |
| [#6](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-SP-ZERO)	|  SP  |  SP  | ZERO |
| [#7](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SP-SP-ONE)	|  SP  |  SP  | ONE  |
|[#8](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SIFT-SIFT-SIFT)| SIFT | SIFT | SIFT |
| [#9](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SIFT-SP-SIFT)	| SIFT |  SP  | SIFT |
|[#10](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SIFT-SP-RAND)	| SIFT |  SP  | RAND |
|[#11](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SIFT-SP-ZERO)	| SIFT |  SP  | ZERO |
|[#12](https://github.com/Xylon-Sean/Detector-oblivious-keypoint-matcher/tree/SIFT-SP-ONE)	| SIFT |  SP  | ONE  |

---

Confirm your current code is in branch **SP-SP-RAND** version.

Run command

```bash
git status | head -1
```

Make sure you are `On branch SP-SP-RAND`.

<img src="assets/Branch-SP-SP-RAND.png" >

Then, run command below in the terminal to get results:

```bash
CUDA_VISIBLE_DEVICES=0 python match_pairs.py --data mega
CUDA_VISIBLE_DEVICES=0 python match_pairs.py --data yfcc
CUDA_VISIBLE_DEVICES=0 python match_pairs.py --data scannet
CUDA_VISIBLE_DEVICES=0 python match_pairs.py --data sun3d
```
