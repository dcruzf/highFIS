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
| Cui, Wu and Xu (2021) | TSK, LogTSK, HTSK | Vowel, Biodeg | [HTSK](htsk.md) |

How to read the comparisons:

- The articles report a mean over random splits that are not published, so a difference
  of the size of the spread between repetitions is expected.
- A reproduction is limited to the datasets that can be obtained with the same samples
  and features as in the article.
- Where highFIS departs from the article, the page says so.
