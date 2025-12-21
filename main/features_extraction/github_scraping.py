import os
from datetime import datetime
import requests

from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv("/.env")
os.getenv("GITHUB_TOKEN")


class GitHubInfo:
    def __init__(self, repo_name):
        self.repo_name = repo_name
        self.github_url = f"https://github.com/{self.repo_name}"
        self.api_url = f"https://api.github.com/repos/{self.repo_name}"
        self.main_page_soup = self.__get_main_page_soup()

    def get_all_features_dict(self):
        primary_language, other_languages = self.get_most_used_languages()
        result = {
            "repo": self.repo_name,
            "num_of_contributors_repo": self.get_num_of_contributors_via_github_page(),
            "primary_language_repo": primary_language,
            "other_languages_repo": other_languages,
            "number_of_files_in_repo": self.get_number_of_files(),
            "created_n_months_ago": self.get_months_old(),
            "latest_commit_in_repo_n_days_ago": self.get_latest_commit(),
            "repo_size_in_kb": self.get_size_in_kb(),
            "num_of_stars_repo": self.get_num_of_stars(),
        }
        return result

    @staticmethod
    def __get_response_from_api(url, **params):
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN")}",
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()

    # ---- web scraping part -------------------------------------------------------------------------------------------

    def __get_main_page_soup(self):
        response = requests.get(self.github_url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup

    def __get_sidebar_with_repo_stats(self):
        return self.main_page_soup.find("div", {"class": "Layout-sidebar"})

    def get_num_of_contributors_via_github_page(self):

        sidebar = self.__get_sidebar_with_repo_stats()
        contributors_link = sidebar.find("a", href=lambda h: h and "contributors" in h)
        num_elem = contributors_link.find("span", class_="Counter")

        num_of_contributors = num_elem.text
        num_of_contributors = int(num_of_contributors.replace(",", ""))
        return num_of_contributors

    # ------------------------------------------------------------------------------------------------------------------

    def get_num_of_contributors_via_api(self, repo_name: str, anon: bool):
        """
        The number may not be the same as on the main GitHub page. The reason here:
        https://github.com/orgs/community/discussions/24355
        :param repo_name:
        :param anon: consider anonymous users (boolean value)
        :return: number of contributors
        """

        def get_url_for_contributors():
            return f"{self.api_url}/contributors?per_page=100&page={page_num}{anon_str}"

        page_num = 1
        anon_str = "&anon=1" if anon else ""
        response = self.__get_response_from_api(get_url_for_contributors())
        num_contributors = 0

        while len(response) > 0:
            num_contributors += len(response)
            page_num += 1
            response = self.__get_response_from_api(get_url_for_contributors())

        return num_contributors

    def get_languages_distribution_dict(self):
        url = self.api_url + "/languages"
        response = self.__get_response_from_api(url)
        sum_lines = sum(v for k, v in response.items())

        language_dict = {k: round(v / sum_lines, 3) for k, v in response.items()}
        return language_dict

    def get_most_used_languages(self):
        languages_dict = self.get_languages_distribution_dict()
        languages_dict_sorted = sorted(
            languages_dict.items(), key=lambda item: item[1], reverse=True
        )
        main_language = languages_dict_sorted[0][0]
        secondary_languages = [
            k for k, v in languages_dict_sorted if v >= 0.1
        ]  # do not consider languages that have less than 10%
        if main_language in secondary_languages:
            secondary_languages.remove(main_language)
        return main_language, secondary_languages

    def get_number_of_files(self):
        url = f"{self.api_url}/git/trees/HEAD"
        data = self.__get_response_from_api(url, recursive=1)
        return sum(1 for e in data["tree"] if e["type"] == "blob")

    def get_months_old(self):
        resp_dict = self.__get_response_from_api(self.api_url)
        datetime_creation_data = datetime.strptime(
            resp_dict["created_at"], "%Y-%m-%dT%H:%M:%SZ"
        )
        now = datetime.now()
        months = (now.year - datetime_creation_data.year) * 12 + (
            now.month - datetime_creation_data.month
        )
        return months

    def get_latest_commit(self):
        url = self.api_url + f"/branches/master"
        data = self.__get_response_from_api(url)
        date = data["commit"]["commit"]["author"]["date"]
        datetime_creation_data = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.now()
        days_delta = (now - datetime_creation_data).days
        return days_delta

    def get_size_in_kb(self):
        return self.__get_response_from_api(self.api_url)["size"]

    def get_num_of_stars(self):
        resp_dict = self.__get_response_from_api(self.api_url)
        return resp_dict["stargazers_count"]

    """Other features to extract:
    - number of files
    """


# if __name__ == "__main__":
#     istio = GitHubInfo("istio/istio")
#     print("ISTIO: contributors ", istio.get_num_of_contributors_via_github_page())
#     print("ISTIO: language_distribution", istio.get_languages_distribution_dict())
#     print("ISTIO: months old", istio.get_months_old())
#     print("ISTIO: latest commit days old ", istio.get_latest_commit())
#     print("ISTIO: size (i kB)", istio.get_size_in_kb())
#     print("ISTIO: number of stars", istio.get_num_of_stars())
#     print("ISTIO: number of files", istio.get_number_of_files())
#
#     numpy_info = GitHubInfo("numpy/numpy")
#     print("NUMPY contributors: ", numpy_info.get_num_of_contributors_via_github_page())
#     print("NUMPY: language_distribution", numpy_info.get_languages_distribution_dict())
#     print("NUMPY: months old", numpy_info.get_months_old())
#     print("NUMPY: latest commit days old", numpy_info.get_latest_commit())
#     print("NUMPY: size (i kB)", numpy_info.get_size_in_kb())
#     print("NUMPY: number of stars", numpy_info.get_num_of_stars())
#     print("NUMPY: number of files", numpy_info.get_number_of_files())
