from subprocess import check_output

print('hello')
result = check_output(['java', '-jar', '--enable-preview', 'Integration/Integration.jar'])
print(result)