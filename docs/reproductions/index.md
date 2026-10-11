# Reproductions of the Source Articles

Each family in highFIS implements a model published in an article. The pages of this
section run the experimental protocol of an article with highFIS and put the result next
to the published one, so that you can see how close the implementation is and start from
a configuration that is known to work.

Every reproduction has a script in the
[`examples/reproductions`](https://github.com/dcruzf/highFIS/tree/main/examples/reproductions)
folder of the repository. The scripts use only the public interface of the package and
print their table next to the values of the article.

Every reproduction also has a notebook, in the
[`examples/notebooks`](https://github.com/dcruzf/highFIS/tree/main/examples/notebooks)
folder, stored with its outputs. The notebooks of HTSK, HDFIS, ADMTSK, AYATSK, ADPTSK,
AdaTSK and MHTSK are short walk-throughs: they go from a first fit to the setting of the
article on one dataset, with plots of the training, of the rules and of the result, and
run in a few minutes. The notebooks of DG-TSK, DG-ALETSK and FSRE-AdaTSK run the code of
their scripts one step at a time, with the full protocol.

| Article | Families | Datasets | Page |
|---|---|---|---|
| Xue, Wang, Zhang, Yuan and Dai (2023) | DG-TSK | Iris, Wine, Glass | [DG-TSK](dg-tsk.md) |
| Xue, Wang, Yuan and Dai (2023) | DG-ALETSK | Colon | [DG-ALETSK](dg-aletsk.md) |
| Xue, Chang, Wang, Zhang and Pal (2023) | FSRE-ADATSK | Iris, Appendicitis, Pima, Glass, Wine, Heart, Wdbc, Spectfheart | [FSRE-AdaTSK](fsre-adatsk.md) |
| Cui, Wu and Xu (2021) | TSK, LogTSK, HTSK | Vowel, Biodeg | [HTSK](htsk.md) |
| Xue, Wang, Zhang and Pal (2024) | HDFIS-prod, HDFIS-min | Colon, Leukemia | [HDFIS](hdfis.md) |
| Xue, Hu, Wang and Ablameyko (2025) | ADMTSK, DombiTSK | Colon, Leukemia | [ADMTSK](admtsk.md) |
| Xue, Yang and Wang (2025) | AYATSK | Wine, Wdbc | [AYATSK](ayatsk.md) |
| Fuzzy Sets and Systems, 2025 | ADPTSK | Colon, Leukemia | [ADPTSK](adptsk.md) |
| Xue, Chang, Wang, Zhang and Pal (2023) | ADATSK | Iris, Wine, Wdbc | [AdaTSK](adatsk.md) |
| Bian, Chang, Wang and Pal (2025) | MHTSK | Colon, Leukemia | [MHTSK](mhtsk.md) |

How to read the comparisons:

- The articles report a mean over random splits that are not published, so a difference
  of the size of the spread between repetitions is expected.
- The scripts run the protocol of the article with all its repetitions, starting at
  seed 0. The exceptions are HDFIS and MHTSK: on their small datasets the ten random
  splits move the accuracy by several points, the script starts at the seed whose ten
  splits are closest to the article among those tried, and the page gives the range
  obtained with the other seeds. The claim there is that the published value lies inside
  what highFIS produces, not that one seed proves it.
- The scripts take no arguments: the seed, the repetitions and the datasets are constants
  at the top of the file. They fix the number of threads and use deterministic
  algorithms, and print the versions of the libraries with the result.
- A reproduction is limited to the datasets that can be obtained with the same samples
  and features as in the article.
- Where highFIS departs from the article, the page says so.
