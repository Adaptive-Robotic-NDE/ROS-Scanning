import os
from setuptools import setup

package_data = {
    "dgpfiles": [
        "*.dpi", # .dpi files for dataguzzler-python
    ]
}

setup(name="dgpfiles",
      description="dgpfiles",
      author="Liam Murray, Tyler Lesthaeghe",
      zip_safe = False,
      packages=["dgpfiles"],
      package_dir={ "dgpfiles": "dgpfiles/utils" },
      package_data = {
          "dgpfiles": [
              "utils/*.dpi", # .dpi files for dataguzzler-python
          ]
      }
    )