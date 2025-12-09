# Release Guide for dpfknn

This guide explains how to release the `dpfknn` package to PyPI.

## Prerequisites

1. **Install build tools:**
   ```bash
   pip install build twine
   ```

2. **PyPI Account:**
   - Create an account at https://pypi.org/
   - Set up two-factor authentication
   - Create an API token at https://pypi.org/manage/account/token/
   - Store the token securely (you'll need it for uploading)

3. **TestPyPI Account (recommended for testing):**
   - Create an account at https://test.pypi.org/
   - Create an API token for testing

## Pre-Release Checklist

Before releasing a new version:

- [ ] Update version number in `pyproject.toml` and `dpfknn/__init__.py`
- [ ] Update `README.md` with any new features or changes
- [ ] Ensure all tests pass: `pytest tests/`
- [ ] Update dependencies if needed
- [ ] Review and update examples if necessary
- [ ] Check that LICENSE file is correct (MIT)
- [ ] Update CHANGELOG if you maintain one

## Build the Package

1. **Clean previous builds:**
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. **Build the distribution packages:**
   ```bash
   python -m build
   ```

   This creates two files in the `dist/` directory:
   - `dpfknn-X.Y.Z-py3-none-any.whl` (wheel distribution)
   - `dpfknn-X.Y.Z.tar.gz` (source distribution)

3. **Check the built distributions:**
   ```bash
   python -m twine check dist/*
   ```

   Note: You may see a warning about "Invalid distribution metadata: unrecognized or malformed field 'license-file'". 
   This is a known issue with newer setuptools versions and can be safely ignored - PyPI will accept the package.

## Test the Package Locally

Before uploading to PyPI, test the package locally:

```bash
# Create a test virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the wheel
pip install dist/dpfknn-*.whl

# Test basic functionality
python -c "from dpfknn import DPFederatedKMeans; print('Import successful!')"

# Run a quick test
python -c "
from dpfknn import DPFederatedKMeans
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
kmeans.fit(X)
print(f'Clustering successful! Inertia: {kmeans.inertia_:.2f}')
"

# Deactivate and remove test environment
deactivate
rm -rf test_env
```

## Upload to TestPyPI (Recommended)

Test the upload process first:

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dpfknn
```

Note: `--extra-index-url` is needed because dependencies come from regular PyPI.

## Upload to PyPI

Once you've verified everything works:

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

## Verify the Release

1. **Check the PyPI page:**
   - Visit https://pypi.org/project/dpfknn/
   - Verify all metadata is correct
   - Check that the README displays properly

2. **Install from PyPI:**
   ```bash
   pip install dpfknn
   ```

3. **Test the installation:**
   ```bash
   python -c "from dpfknn import DPFederatedKMeans; print('Success!')"
   ```

## Post-Release

1. **Tag the release in Git:**
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. **Create a GitHub Release:**
   - Go to https://github.com/Bowenislandsong/Differentially-Private-Federated-KNN/releases
   - Click "Create a new release"
   - Select the tag you just created
   - Add release notes
   - Attach the distribution files from `dist/`

3. **Announce the release:**
   - Update project documentation
   - Announce on relevant channels (if applicable)

## Troubleshooting

### Build Issues

**Problem:** Build fails with missing dependencies
```bash
pip install --upgrade setuptools wheel build
```

**Problem:** Files not included in distribution
- Check `MANIFEST.in` for proper include/exclude patterns
- Run `python -m build` and inspect the `.tar.gz` file:
  ```bash
  tar -tzf dist/dpfknn-*.tar.gz
  ```

### Upload Issues

**Problem:** Authentication fails
- Make sure you're using `__token__` as the username
- Verify your API token is correct and has upload permissions
- Check that 2FA is properly configured

**Problem:** Version already exists
- You cannot overwrite an existing version on PyPI
- Increment the version number in `pyproject.toml` and rebuild

**Problem:** Invalid package metadata
- Run `python -m twine check dist/*` to identify issues
- Common issues: missing required fields, invalid classifiers

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): Backward-compatible new features
- **PATCH** version (0.0.X): Backward-compatible bug fixes

Example progression: `0.1.0` → `0.1.1` → `0.2.0` → `1.0.0`

## Security Notes

1. **Never commit API tokens** to version control
2. **Use API tokens**, not passwords for PyPI uploads
3. **Limit token scope** to only the necessary permissions
4. **Rotate tokens** periodically
5. **Enable 2FA** on your PyPI account

## Configuration File (Optional)

You can create a `~/.pypirc` file to store repository configurations:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

**Warning:** This stores tokens in plain text. Use with caution.

## Continuous Integration

Consider setting up GitHub Actions for automated releases:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Store your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`.

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [setuptools Documentation](https://setuptools.pypa.io/)
