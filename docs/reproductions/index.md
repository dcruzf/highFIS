# Reproductions of the Source Articles

Each family in highFIS implements a model published in an article. The pages of this
section run the experimental protocol of an article with highFIS and put the result next
to the published one, so that you can see how close the implementation is and start from
a configuration that is known to work.

Every reproduction has a script in the
[`examples/reproductions`](https://github.com/dcruzf/highFIS/tree/main/examples/reproductions)
folder of the repository. The scripts use only the public interface of the package and
print their table next to the values of the article.

| Article | Families | Datasets | Page |
|---|---|---|---|
| Xue, Wang, Zhang, Yuan and Dai (2023) | DG-TSK | Iris, Wine | [DG-TSK](dg-tsk.md) |
| Xue, Wang, Yuan and Dai (2023) | DG-ALETSK | Colon | [DG-ALETSK](dg-aletsk.md) |
| Xue, Chang, Wang, Zhang and Pal (2023) | FSRE-ADATSK | Iris, Wine, Wdbc | [FSRE-AdaTSK](fsre-adatsk.md) |
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
- On small datasets the partition moves the accuracy by several points. Where it does,
  the script uses the seed whose result is closest to the article among those tried,
  and the page gives the range obtained with the other seeds. The claim is that the
  published value lies inside what highFIS produces, not that one seed proves it.
- The scripts fix the number of threads, because the result depends on it.
- A reproduction is limited to the datasets that can be obtained with the same samples
  and features as in the article.
- Where highFIS departs from the article, the page says so.
