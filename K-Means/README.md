# K-means
K-MEANS算法是输入聚类个数k，以及包含 n个数据对象的数据库，输出满足方差最小标准k个聚类的一种算法。k-means 算法接受输入量 k ；然后将n个数据对象划分为 k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。                                                          基本步骤                                                                                                                                      （1） 从 n个数据对象任意选择 k 个对象作为初始聚类中心；                                                                                        （2） 根据每个聚类对象的均值（中心对象），计算每个对象与这些中心对象的距离；并根据最小距离重新对相应对象进行划分；                                    （3） 重新计算每个（有变化）聚类的均值（中心对象）；                                                                                            （4） 计算标准测度函数，当满足一定条件，如函数收敛时，则算法终止；如果条件不满足则回到步骤（2）。